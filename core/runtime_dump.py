"""Paired profiling predictions and execution measurements in two run streams."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import logging
import os
from pathlib import Path
import re
from threading import Lock

from core.profile_mask import ProfileMask

logger = logging.getLogger(__name__)
_trace = ContextVar('operator_step_trace', default=None)
_invocation = ContextVar('operator_invocation', default=None)
_execution_tokens = ContextVar('execution_tokens', default=None)
_token_weight = ContextVar('token_weight', default=1)
_step_metadata = ContextVar('operator_step_metadata', default={})
_default_dumper = None
_dumper_lock = Lock()


def tracing_enabled():
    return _trace.get() is not None


def observe_mask(op, mask, input_mask=None, output_mask=None):
    call = _invocation.get()
    if call is not None and call['oid'] == op.oid:
        call['mask'] = mask
        if input_mask is not None:
            call['input_masks'] = [input_mask]
            call['output_masks'] = [output_mask] if output_mask is not None else []
            call['input_rows'] = input_mask.count
            call['output_rows'] = output_mask.count if output_mask is not None else 0
            call['io_observed'] = True


def observe_tokens(prefill, decode, latency=0.0):
    meter = _execution_tokens.get()
    if meter is not None:
        meter[0] += prefill or 0
        meter[1] += decode or 0
        meter[2] += (prefill or 0) + _token_weight.get() * (decode or 0)
    call = _invocation.get()
    if call is not None:
        call['prefill'] += prefill or 0
        call['decode'] += decode or 0
        # Round trips and their wall time, for the per-call overhead model.
        if prefill:
            call['model_calls'] = call.get('model_calls', 0) + 1
            call['latency'] = call.get('latency', 0.0) + (latency or 0.0)


@contextmanager
def measure_execution_tokens():
    """Count model tokens without enabling operator tracing or dump files."""
    meter = [0, 0, 0]
    token = _execution_tokens.set(meter)
    try:
        yield meter
    finally:
        _execution_tokens.reset(token)


def observe_evaluations(count):
    """Count new model prompts, excluding deduplicated or cached source rows."""
    call = _invocation.get()
    if call is not None:
        call['evaluation_count'] += count


def _batch_mask(batch):
    if batch is None or not batch.tables:
        return None
    try:
        return ProfileMask.from_batch(batch)
    except (IndexError, TypeError, ValueError):
        logger.exception('Cannot represent batch lineage in an operator dump')
        return None


def _domain(mask):
    return {'tables': mask.tables, 'row_ids': mask.rows}


def _mask_document(mask):
    if mask is None:
        return None
    return {**_domain(mask), 'bitmap': [bool(bit) for bit in mask.bitmap]}


def _cost_fields(prefill, decode, weight):
    return {'prefill_tokens': prefill, 'decode_tokens': decode,
            'decode_weight': weight, 'prefill_cost': prefill,
            'decode_cost': weight * decode, 'token_cost': prefill + weight * decode}


def _age(sample, metadata):
    query_step = metadata.get('query_step')
    sample_query_step = sample.get('sample_query_step')
    if query_step is not None and sample_query_step is not None:
        return query_step - sample_query_step, 'query_steps'
    step, sample_step = metadata.get('profile_step'), sample.get('sample_step')
    return (step - sample_step if step is not None and sample_step is not None else None), 'profile_steps'


def trace_operator(func):
    """Capture logical input/output identities and actual new LLM work."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        weight_token = _token_weight.set(
            10 if self.__type_id__.name == 'SEM_PROJECTION' else 1)
        trace = _trace.get()
        if trace is None:
            try:
                return func(self, *args, **kwargs)
            finally:
                _token_weight.reset(weight_token)
        payload = args[0] if args else kwargs.get('payload')
        kind = self.__type_id__.name
        is_profile = kwargs.get('is_profile', False)
        if kind not in {'JOIN', 'SEM_JOIN'} and len(args) > 1:
            is_profile = args[1]
        before = _batch_mask(payload)
        previous_sample = getattr(self, '_profile_sample', None)
        input_masks = [before] if before is not None else []
        input_rows = len(payload.data) if payload is not None else 0
        if payload is not None and payload.is_eof and isinstance(getattr(self, 'buffer', None), list):
            buffered = self.buffer
            input_rows = sum(len(batch.data) for batch in buffered)
            input_masks = [mask for batch in buffered if (mask := _batch_mask(batch)) is not None]
        call = {'oid': self.oid, 'prefill': 0, 'decode': 0, 'evaluation_count': 0,
                'input_rows': input_rows, 'output_rows': 0,
                'input_masks': input_masks, 'output_masks': [], 'status': 'started',
                'already_applied': payload is not None and self.oid in payload.fingerprint}
        trace.calls.setdefault(self.oid, []).append(call)
        token = _invocation.set(call)
        try:
            result = func(self, *args, **kwargs)
            after = _batch_mask(result)
            if not call.get('io_observed'):
                call['output_rows'] = len(result.data) if result is not None else 0
                call['output_masks'] = [after] if after is not None else []
            call['status'] = 'buffered' if result is None else ('eof' if result.is_eof else 'completed')
            sample = getattr(self, '_profile_sample', None)
            if is_profile and sample is not None and sample is not previous_sample:
                call['mask'] = sample['mask']
                if kind in {'FILTER', 'SEM_FILTER'} and before is not None:
                    # Profile-mode filters return their unfiltered input. Log
                    # their measured output rather than that transport batch.
                    measured = before.intersect(sample['mask'])
                    call['output_masks'] = [measured]
                    call['output_rows'] = measured.count
            elif 'mask' not in call and result is not None and not result.is_eof:
                if kind in {'FILTER', 'SEM_FILTER'} and before is not None and after is not None:
                    kept = set(after.rows)
                    call['mask'] = ProfileMask(before.tables, before.rows,
                                               tuple(row in kept for row in before.rows))
                elif kind not in {'JOIN', 'SEM_JOIN'} and after is not None:
                    call['mask'] = after
            if kind in {'JOIN', 'SEM_JOIN'} and not call.get('io_observed'):
                # Feeding a join's left buffer or an EOF marker evaluates no pairs.
                call.update(input_rows=0, output_rows=0, input_masks=[], output_masks=[])
            elif 'mask' not in call and before is not None:
                call['mask'] = before
            if is_profile:
                call['sample_step'] = trace.metadata.get('profile_step')
                call['sample_query_step'] = trace.metadata.get('query_step')
                self._dump_profile_record = call.copy()
            return result
        except Exception:
            call['status'] = 'error'
            raise
        finally:
            _invocation.reset(token)
            _token_weight.reset(weight_token)
    return wrapper


class StepTrace:
    def __init__(self, phase, operators, metadata=None):
        self.phase = phase
        self.operators = {op.oid: op for op in operators}
        self.metadata = {**_step_metadata.get(), **(metadata or {})}
        self.calls = {}
        self.total_cost = None
        self.prediction = None
        self.candidate_costs = None
        self.candidate_operator_catalog = None
        self.call_model = None

    def document(self):
        operators = {}
        for oid, op in self.operators.items():
            calls = self.calls.get(oid, [])
            sample = getattr(op, '_profile_sample', None)
            records = calls
            if self.phase == 'profile':
                retained = getattr(op, '_dump_profile_record', None)
                records = [retained] if retained is not None else []
            masks = [call['mask'] for call in records if 'mask' in call]
            prefill = sum(call['prefill'] for call in records)
            decode = sum(call['decode'] for call in records)
            input_rows = sum(call['input_rows'] for call in records)
            output_rows = sum(call['output_rows'] for call in records)
            evaluations = sum(call['evaluation_count'] for call in records)
            if self.phase == 'profile' and sample is not None:
                masks = [sample['mask']]
                prefill, decode = sample.get('prefill', 0) or 0, sample.get('decode', 0) or 0
                input_rows = sample.get('input_sz', sample.get('input_left_sz', 0) * sample.get('input_right_sz', 0))
                output_rows = sample.get('output_sz', sample['mask'].count)
                evaluations = sample.get('work', 0) if op.__type_id__.name.startswith('SEM_') else 0
            entry = {
                'op_name': getattr(op, 'op_name', None), 'op_type': op.__type_id__.name,
                'bitmap': [bool(bit) for mask in masks for bit in mask.bitmap],
                'bitmap_domains': [_domain(mask) for mask in masks],
                'input_rows': input_rows, 'output_rows': output_rows,
                'evaluation_count': evaluations,
                'evaluation_unit': ('source_rows' if op.__type_id__.name in {'SEM_FILTER', 'SEM_PROJECTION'}
                                    else 'tuple_pairs' if op.__type_id__.name in {'SEM_JOIN', 'SEM_SORT'}
                                    else 'prompts'),
                'input_domains': [_mask_document(mask) for call in records for mask in call['input_masks']],
                'output_domains': [_mask_document(mask) for call in records for mask in call['output_masks']],
                **_cost_fields(prefill, decode, 10 if op.__type_id__.name == 'SEM_PROJECTION' else 1),
            }
            if self.phase == 'profile':
                stamp = sample if sample is not None else (records[0] if records else {})
                age, unit = _age(stamp, self.metadata)
                entry.update(sample_step=stamp.get('sample_step'), sample_age=age,
                             sample_age_unit=unit,
                             sample_origin=stamp.get('sample_origin', 'profile'))
            else:
                entry['calls'] = [
                    {key: call[key] for key in ('input_rows', 'output_rows', 'evaluation_count', 'status')}
                    for call in calls
                ]
                # Round trips and their wall time, comparable with the
                # predictions' request waves and call cost.
                from core.llm import request_waves
                entry.update(
                    model_calls=sum(call.get('model_calls', 0) for call in calls),
                    request_waves=sum(request_waves(call['evaluation_count'], call['model_calls'])
                                      or call['model_calls']
                                      for call in calls if call.get('model_calls')),
                    latency_sec=sum(call.get('latency', 0.0) for call in calls))
            operators[str(oid)] = entry
        total_prefill = sum(op['prefill_cost'] for op in operators.values())
        total_decode = sum(op['decode_cost'] for op in operators.values())
        document = {**self.metadata, 'operators': operators}
        if self.prediction is not None:
            predictions = {}
            for oid, details in self.prediction.items():
                fields = {key: value for key, value in details.items() if key not in {'mask', 'input_mask'}}
                fields['input_domain'] = _mask_document(details['input_mask'])
                fields['output_domain'] = _mask_document(details['mask'])
                fields['output_rows'] = fields.pop('output_sz')
                predictions[str(oid)] = fields
            document['predictions'] = predictions
            total_prefill = sum(op['amortized_prefill_cost'] for op in predictions.values())
            total_decode = sum(op['amortized_decode_cost'] for op in predictions.values())
        if self.candidate_costs is not None:
            document['candidate_costs'] = self.candidate_costs
            document['candidate_cost_fields'] = [
                'input_rows', 'output_rows', 'evaluation_count',
                'token_cost', 'amortization_factor', 'amortized_token_cost']
            document['candidate_operator_catalog'] = self.candidate_operator_catalog
        if self.call_model is not None:
            document['call_model'] = self.call_model
        if self.phase == 'execution':
            document.update(
                total_request_waves=sum(op['request_waves'] for op in operators.values()),
                total_latency_sec=sum(op['latency_sec'] for op in operators.values()))
        document.update(total_prefill_cost=total_prefill, total_decode_cost=total_decode,
                        total_cost=self.total_cost if self.total_cost is not None else total_prefill + total_decode)
        return document


class StepDumper:
    def __init__(self, directory=None, enabled=True, run_id='default'):
        self.enabled = enabled
        root = Path(directory) if directory is not None else Path(__file__).resolve().parents[1] / 'op_dumps_t'
        self.directory = root
        self.run_id = re.sub(r'[^A-Za-z0-9_.-]+', '-', run_id)
        self.counters = {'profile': 0, 'execution': 0}
        self.lock = Lock()
        self.initialized = False

    def path(self, phase):
        return self.directory / f'{self.run_id}_{phase}.jsonl'

    def start(self):
        if not self.enabled:
            return
        with self.lock:
            if not self.initialized:
                self.directory.mkdir(parents=True, exist_ok=True)
                for phase in self.counters:
                    self.path(phase).write_text('')
                self.initialized = True
                logger.info('Operator dumps: %s_[profile|execution].jsonl',
                            self.directory / self.run_id)

    @contextmanager
    def step(self, phase, operators, capture=False, **metadata):
        if not self.enabled and not capture:
            yield None
            return
        trace = StepTrace(phase, operators, metadata)
        token = _trace.set(trace)
        try:
            yield trace
        finally:
            _trace.reset(token)
            if self.enabled:
                try:
                    document = trace.document()
                    self.start()
                    with self.lock:
                        self.counters[phase] += 1
                        document = {'step': self.counters[phase], **document}
                        with self.path(phase).open('a') as stream:
                            stream.write(json.dumps(document, default=_json_value) + '\n')
                except (OSError, TypeError, ValueError):
                    logger.exception('Could not write operator step dump to %s', self.directory)


def get_dumper():
    global _default_dumper
    with _dumper_lock:
        if _default_dumper is None:
            _default_dumper = StepDumper(
                os.environ.get('SEM_QUERY_DUMP_DIR'),
                enabled=os.environ.get('SEM_QUERY_DUMP', '1') != '0')
        return _default_dumper


def _configure_dumper(run_id):
    global _default_dumper
    with _dumper_lock:
        _default_dumper = StepDumper(os.environ.get('SEM_QUERY_DUMP_DIR'),
                                    enabled=os.environ.get('SEM_QUERY_DUMP', '1') != '0',
                                    run_id=run_id)
        try:
            _default_dumper.start()
        except OSError:
            logger.exception('Could not initialize operator dumps')
        return _default_dumper


def _fluctuation_tag(randomize):
    size = randomize.get_batch_size()
    tag = '' if size == randomize.DEFAULT_BATCH_SIZE else f'_bs{size}'
    return _schedule_tag(randomize) + tag


def _schedule_tag(randomize):
    mode, percent = randomize.FLUCTUATION_MODE, randomize.FLUCTUATION_PERCENT
    if mode == 'regime':
        return f'_regime{randomize.REGIME_SEGMENTS}_pct{percent:g}'
    if mode == 'crossover':
        return '_crossover'
    if mode == 'burst':
        return f'_burst{randomize.FLUCTUATION_BURSTS}_pct{percent:g}'
    return f'_fluc{percent:g}'


def configure_experiment_dump(dataset, plans, n_queries, mode,
                              full_batch_execution=False, seed=None):
    """Give each experiment mode its own pair of execution/profile streams."""
    from config import randomize

    if mode not in {'static', 'pd', 'pu'}:
        raise ValueError(f'Unknown experiment mode: {mode}')
    plan_tag = '-'.join(Path(path).stem for path in plans)[:60]
    seed_tag = f'_seed{seed}' if seed is not None else ''
    batch_tag = '_fullbatch' if full_batch_execution else ''
    fluctuation_tag = _fluctuation_tag(randomize)
    run_id = (f'{dataset}_{plan_tag}{seed_tag}_cfg{int(randomize.get_use_config())}'
              f'_sel{randomize.DEFAULT_SEL_VARIANCE:g}'
              f'{fluctuation_tag}_q{n_queries}'
              f'_{mode}{batch_tag}_luna')
    return _configure_dumper(run_id)


def configure_build_dump(dataset, plans, seed, lam, n_queries, llm, slm, mode=None,
                         full_batch_execution=False, fully_pipelined=False,
                         pipeline_profile_interval=1):
    """Reset exactly two streams for the same build inputs before workers start."""
    from config import randomize

    settings = {key: value for key, value in vars(randomize).items()
                if key.isupper() and isinstance(value, (str, int, float, bool, list, tuple, dict))}
    spec = {
        'dataset': dataset, 'plans': [(Path(p).stem, Path(p).read_text()) for p in plans],
        'seed': seed, 'config_seed': randomize.get_config_seed(),
        'lambda': lam, 'queries': n_queries, 'settings': settings,
        'full_batch_execution': full_batch_execution,
        'fully_pipelined': fully_pipelined,
        'pipeline_profile_interval': pipeline_profile_interval,
        'models': [{key: getattr(model, key, None) for key in ('model_name', 'url', 'timeout')}
                   for model in (llm, slm)],
    }
    plan_tag = '-'.join(Path(p).stem for p in plans)[:60]
    mode_tag = f'_{mode}' if mode else ''
    batch_tag = '_fullbatch' if full_batch_execution else ''
    fluctuation_tag = _fluctuation_tag(randomize)
    pipeline_tag = (f'_pipeline_every{pipeline_profile_interval}'
                    if fully_pipelined and pipeline_profile_interval != 1 else
                    '_pipeline' if fully_pipelined else '')
    run_id = (f'{dataset}_{plan_tag}_seed{seed}_cfg{int(randomize.get_use_config())}'
              f'_sel{randomize.DEFAULT_SEL_VARIANCE:g}{fluctuation_tag}'
              f'_q{n_queries}{mode_tag}{batch_tag}{pipeline_tag}_luna')
    return _configure_dumper(run_id)


def _json_value(value):
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError(f'Cannot serialize {type(value).__name__}')


def dump_query_step(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        token = _step_metadata.set({'query_id': self.qid, 'query_step': self.completed_step + 1,
                                    'dataset': self.db_name, 'plan_idx': self.plan_idx})
        try:
            return func(self, *args, **kwargs)
        finally:
            _step_metadata.reset(token)
    return wrapper


@contextmanager
def profile_step_metadata(query, step):
    """Label a speculative profile with the query step it prepares."""
    token = _step_metadata.set({'query_id': query.qid, 'query_step': step,
                                'dataset': query.db_name, 'plan_idx': query.plan_idx})
    try:
        yield
    finally:
        _step_metadata.reset(token)


def dump_execution(func):
    @wraps(func)
    def wrapper(self, op_oid_map, is_profile=False, rollback=False):
        if is_profile or tracing_enabled():
            return func(self, op_oid_map, is_profile=is_profile, rollback=rollback)
        link = dict(getattr(self, '_dump_prediction_link', {}))
        age, unit = _age(link, {**link, **_step_metadata.get()})
        link.update(sample_age=age, sample_age_unit=unit,
                    plan={'root_oid': self.root, 'children': self.ch_link},
                    full_batch_execution=self.full_batch_execution)
        with get_dumper().step('execution', self._get_ops(op_oid_map),
                               capture=(self.full_batch_execution or self.capture_execution_feedback),
                               **link) as trace:
            result = func(self, op_oid_map, is_profile=is_profile, rollback=rollback)
            self._last_execution_trace = trace
            return result
    return wrapper


def dump_profile(func):
    @wraps(func)
    def wrapper(self, candidate_subqueries, op_oid_map, profile_enabled=True):
        subquery = self._get_profile_subq(candidate_subqueries)
        profile_step = self.step_counter + 1
        with get_dumper().step('profile', subquery._get_ops(op_oid_map), profile_step=profile_step) as trace:
            result = func(self, candidate_subqueries, op_oid_map, profile_enabled=profile_enabled)
            if trace is not None:
                sample_step = candidate_subqueries._dump_sample_step
                if sample_step == profile_step:
                    candidate_subqueries._dump_sample_query_step = trace.metadata.get('query_step')
                stamp = {'sample_step': sample_step,
                         'sample_query_step': getattr(candidate_subqueries, '_dump_sample_query_step', None)}
                age, unit = _age(stamp, trace.metadata)
                selected = candidate_subqueries.subqueries[result[0]]
                link = {**stamp, 'profile_step': profile_step, 'selected_candidate_id': result[0]}
                selected._dump_prediction_link = link
                trace.metadata.update(**link, sample_age=age, sample_age_unit=unit,
                                      plan={'root_oid': selected.root, 'children': selected.ch_link},
                                      profiled_plan={'root_oid': subquery.root, 'children': subquery.ch_link})
                for op in trace.operators.values():
                    sample = getattr(op, '_profile_sample', None)
                    if sample is not None and sample.get('sample_step') == profile_step:
                        sample['sample_query_step'] = trace.metadata.get('query_step')
                trace.total_cost = result[1]
                trace.prediction = self._last_prediction
                overhead_sec, sec_per_cost = getattr(self, '_call_fit', (None, None))
                trace.call_model = {'overhead_cost': self.call_overhead_cost,
                                    'overhead_sec': overhead_sec,
                                    'sec_per_cost': sec_per_cost}
                trace.candidate_costs = self._candidate_breakdowns
                trace.candidate_operator_catalog = {
                    str(oid): {
                        'op_name': getattr(op_oid_map[oid], 'op_name', None),
                        'op_type': op_oid_map[oid].__type_id__.name,
                    }
                    for oid in {oid for candidate in self._candidate_breakdowns
                                for oid in map(int, candidate['operators'])}
                }
            return result
    return wrapper
