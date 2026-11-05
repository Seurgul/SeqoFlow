from collections import deque
from math import ceil, isclose
from random import Random
from typing import List, Dict, Optional
import logging

from core.llm import VLLMWrapper, request_waves
from core.runtime_dump import dump_profile, tracing_enabled
from core.base import SubQuery, SubQueryGroup, Operator, TYPE_ID
from core.profile_mask import ProfileMask

logger = logging.getLogger(__name__)
PROFILE_REFRESH_INTERVAL = 15
PROFILE_MAX_SAMPLE_AGE = 15
# DRIFT_MIN_RATE_CHANGE = 0.2
# DRIFT_STANDARD_ERRORS = 2


class _ProfileBudget:
    """Limit model prompts in one profile refresh, across all operators."""

    def __init__(self, limit):
        self.limit = limit
        self.used = 0


class _BudgetedModel:
    def __init__(self, model, budget, allowance, seed):
        self.model = model
        self.budget = budget
        self.allowance = allowance
        self.seed = seed
        self.calls = 0
        self.used = 0
        self.pair_shape = None

    def set_pair_shape(self, left, right):
        self.pair_shape = (left, right)

    def as_llm_call(self, prompts, **kwargs):
        count = len(prompts)
        if not count:
            self.pair_shape = None
            return self.model.as_llm_call(prompts, **kwargs)
        take = min(count, self.allowance, self.budget.limit - self.budget.used)
        if not take:
            raise ValueError("Profile budget cannot give every semantic operator a sample")
        rng = Random(self.seed + self.calls)
        self.calls += 1
        shape, self.pair_shape = self.pair_shape, None
        if shape is not None and shape[0] * shape[1] == count and take < count:
            left, right = shape
            left_start, right_start = rng.randrange(left), rng.randrange(right)
            positions = {((left_start + k * left // take) % left) * right +
                         ((right_start + k * right // take) % right)
                         for k in range(take)}
            if len(positions) < take:
                positions.update(rng.sample([i for i in range(count) if i not in positions],
                                            take - len(positions)))
            positions = sorted(positions)
        else:
            positions = sorted(rng.sample(range(count), take))
        measured = self.model.as_llm_call([prompts[i] for i in positions], **kwargs)
        responses = measured['responses']
        if len(responses) != take:
            raise ValueError("Profile model returned an unexpected number of responses")
        self.budget.used += take
        self.used += take
        self.allowance -= take
        if take == count:
            return measured

        filled = list(responses)
        by_position = dict(zip(positions, responses))
        answers = [by_position[i] if i in by_position else rng.choice(filled)
                   for i in range(count)]
        return {**measured, 'responses': answers}


class CostModel:
    """
    This class is used to compute the virtual cost of an operator based on profile data
    """
    def __init__(self):
        self.execution_context = {}
        self.observations = {}
        self.call_overhead_cost = 0.0

    def _observation(self, op, inputs):
        """Use execution evidence only for the same source rows or join pairs."""
        if inputs is None:
            return None
        rows = set(inputs.surviving_rows())
        for observed in reversed(self.observations.get(op.oid, ())):
            domain = observed['input_mask']
            if domain.tables == inputs.tables and rows.issubset(
                    set(domain.surviving_rows())):
                return observed
        return None

    @staticmethod
    def _restrict(mask, retained):
        """Keep only tuples still present in a live buffer, including correlations."""
        indices = [mask.tables.index(table) for table in retained.tables]
        present = set(retained.surviving_rows())
        return ProfileMask(mask.tables, mask.rows, tuple(
            bool(keep and tuple(row[i] for i in indices) in present)
            for row, keep in zip(mask.rows, mask.bitmap)))

    def _unary_cache(self, op, inputs):
        indices = [inputs.tables.index(table) for table in op.tables]
        if 'unary_known' in self.execution_context:
            return indices, self.execution_context['unary_known'].get(op.oid, {})
        known = {}
        for retained in self.execution_context.get('completed', {}).get(op.oid, []):
            projected = retained.project(op.tables)
            known.update((row, True) for row in projected.surviving_rows())
        if len(op.tables) == 1:
            for row in inputs.rows:
                identity = (row[indices[0]],)
                batch, position = identity[0]
                decisions = getattr(op, 'cache_bm', {}).get(batch, {})
                if position in decisions:
                    known[identity] = bool(decisions[position])
                elif position in getattr(op, 'bms', {}).get(batch, set()):
                    known[identity] = False
                if op.__type_id__ == TYPE_ID.SEM_PROJECTION:
                    cached = getattr(op, 'cache_', {}).get(batch)
                    if cached is not None:
                        found = cached[cached['row_id'] == position]['new_value']
                        if not found.empty and found.notna().all():
                            known[identity] = True
        return indices, known

    def get_cost_op(self, op: Operator, upstream_info=None):
        """Compose sampled tuple masks, then charge for the surviving inputs.

        Unary semantic operators evaluate distinct source rows; semantic joins
        evaluate every input tuple pair. Predicate selectivities are diagnostic
        only and never multiplied when propagating cardinality.
        """
        profile = op._get_profile() or {}
        op_type = op.__type_id__
        semantic = op_type in {TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION, TYPE_ID.SEM_JOIN}
        decode_weight = 10 if op_type == TYPE_ID.SEM_PROJECTION else 1

        def finish(inputs, output, work=0, observation=None):
            sample = (observation if observation is not None and
                      observation.get('work', 0) else profile)
            sampled_work = sample.get('work', 0)
            scale = work / sampled_work if semantic and sampled_work else 0.0
            prefill_tokens = (sample.get('prefill', 0) or 0) * scale
            decode_tokens = (sample.get('decode', 0) or 0) * scale
            prefill_cost, decode_cost = prefill_tokens, decode_weight * decode_tokens
            # Each wave of concurrent prompts pays one model round trip.
            waves = request_waves(work) if semantic and work else 0
            call_cost = self.call_overhead_cost * waves
            return prefill_cost + decode_cost + call_cost, {
                'mask': output, 'input_mask': inputs,
                'input_rows': inputs.count if inputs is not None else 0,
                'output_sz': output.count, 'evaluation_count': work if semantic else 0,
                'evaluation_unit': ('tuple_pairs' if op_type in {TYPE_ID.SEM_JOIN, TYPE_ID.SEM_SORT}
                                    else 'source_rows' if op_type in {TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION}
                                    else 'prompts'),
                'cost_model_supported': op_type not in {
                    TYPE_ID.SEM_GROUPBY, TYPE_ID.SEM_AGGREGATION, TYPE_ID.SEM_SORT},
                'prefill_tokens': prefill_tokens, 'decode_tokens': decode_tokens,
                'decode_weight': decode_weight, 'prefill_cost': prefill_cost,
                'decode_cost': decode_cost, 'token_cost': prefill_cost + decode_cost,
                'call_cost': call_cost, 'request_waves': waves,
            }

        if op_type in {TYPE_ID.SCAN, TYPE_ID.SOURCE}:
            output = profile['mask']
            if self.execution_context.get('eof'):
                return finish(None, ProfileMask(output.tables, (), ()))
            for retained in self.execution_context.get('leaves', {}).get(op.oid, []):
                output = self._restrict(output, retained)
            return finish(None, output)
        if op_type in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}:
            assert len(upstream_info) == 2, "Join operator must have two children"
            children = [info['mask'] for info in upstream_info]
            for i, retained in enumerate(self.execution_context.get('joins', {}).get(op.oid, (None, None))):
                if retained is not None:
                    children[i] = self._restrict(children[i], retained)
            inputs = children[0].cross(children[1])
            if not inputs.count:
                return finish(inputs, inputs)
            completed = self.execution_context.get('completed', {}).get(op.oid, [])
            if completed:
                output = inputs
                for retained in completed:
                    output = self._restrict(output, retained)
                return finish(inputs, output)
            observed = self._observation(op, inputs)
            mask = observed['mask'] if observed is not None else profile['mask']
            return finish(inputs, inputs.intersect(mask), inputs.count, observed)

        inputs = upstream_info[0]['mask']
        if not inputs.count:
            return finish(inputs, inputs)
        observed = self._observation(op, inputs) if semantic else None
        output = inputs
        indices, known = (self._unary_cache(op, inputs)
                          if op_type in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION}
                          and self.execution_context else (None, {}))
        if op_type in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER}:
            if indices is None:
                output = inputs.intersect(observed['mask'] if observed else profile['mask'])
            else:
                sample_mask = observed['mask'] if observed else profile['mask']
                decisions = dict(zip(sample_mask.rows, sample_mask.bitmap))
                output = ProfileMask(inputs.tables, inputs.rows, tuple(
                    bool(keep and (known[key] if key in known else decisions[key]))
                    for row, keep in zip(inputs.rows, inputs.bitmap)
                    for key in [tuple(row[i] for i in indices)]))
        if op_type in {TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION}:
            work = (inputs.distinct_count(op.tables) if indices is None else len({
                tuple(row[i] for i in indices) for row in inputs.surviving_rows()
                if tuple(row[i] for i in indices) not in known}))
            return finish(inputs, output, work, observed)
        if op_type in {
            TYPE_ID.FILTER, TYPE_ID.PROJECTION, TYPE_ID.SORT, TYPE_ID.AGGREGATION,
            TYPE_ID.SEM_GROUPBY, TYPE_ID.SEM_AGGREGATION, TYPE_ID.SEM_SORT, TYPE_ID.SINK,
        }:
            return finish(inputs, output)
        raise NotImplementedError(f"Cost model for operator type {op_type} is not implemented.")


class Profiler:
    def __init__(self, profile_model: VLLMWrapper, profile_budget: int = 20):
        if profile_budget < 1:
            raise ValueError("profile_budget must be positive")
        self.model = profile_model
        self.profile_budget = profile_budget
        self.last_profile_evaluations = 0
        self.cost_model = CostModel()
        self._observations = {}
        self.cost_model.observations = self._observations
        self.step_counter = 0
        self._call_samples = deque(maxlen=64)
        self.call_overhead_cost = None
        self._call_fit = (None, None)
        self._executed_rates = {}
        self.last_step_cost = None

    def _fit_call_overhead(self):
        """Least-squares fit of seconds = overhead + rate * token cost."""
        if len(self._call_samples) < 8:
            return
        costs = [cost for cost, _ in self._call_samples]
        seconds = [sec for _, sec in self._call_samples]
        mean_cost, mean_sec = sum(costs) / len(costs), sum(seconds) / len(seconds)
        var = sum((c - mean_cost) ** 2 for c in costs)
        if var <= 0:
            return
        rate = sum((c - mean_cost) * (s - mean_sec) for c, s in zip(costs, seconds)) / var
        overhead = max(0.0, mean_sec - rate * mean_cost)
        # A flat or noisy fit means calls cost the same regardless of size, bound the overhead by the largest observed call instead.
        estimate = overhead / rate if rate > 0 else max(costs)
        estimate = min(estimate, max(costs))
        self._call_fit = (overhead, rate)
        if self.call_overhead_cost is None:
            logger.info('Calibrated call overhead: %.0f token cost (%.3fs + %.6fs per token cost)',
                        estimate, overhead, rate)
        self.call_overhead_cost = estimate

    def observe_execution(self, trace):
        """Retain actual per-batch masks and token rates without model calls."""
        if trace is None:
            return False
        drift = False
        weights = {oid: 10 if op.__type_id__ == TYPE_ID.SEM_PROJECTION else 1
                   for oid, op in trace.operators.items()}
        step_cost = sum(call['prefill'] + weights.get(oid, 1) * call['decode']
                        for oid, calls in trace.calls.items() for call in calls)
        for oid, calls in trace.calls.items():
            for call in calls:
                if call.get('model_calls'):
                    cost = call['prefill'] + weights.get(oid, 1) * call['decode']
                    waves = (request_waves(call['evaluation_count'], call['model_calls'])
                             or call['model_calls'])
                    self._call_samples.append((cost / waves, call['latency'] / waves))
        self._fit_call_overhead()
        self.last_step_cost = step_cost
        for oid, calls in trace.calls.items():
            op = trace.operators[oid]
            if op.__type_id__ not in {
                    TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION, TYPE_ID.SEM_JOIN}:
                continue
            for call in calls:
                if call['status'] != 'completed' or not call.get('input_masks') or 'mask' not in call:
                    continue
                domain = call['input_masks'][0]
                if domain is None or not domain.count:
                    continue
                sample = op._get_profile() or {}
                sample_mask = sample.get('mask')
                sample_work = sample.get('work', 0)
                work = call['evaluation_count']
                weight = 10 if op.__type_id__ == TYPE_ID.SEM_PROJECTION else 1
                actual_cost = call['prefill'] + weight * call['decode']
                sample_cost = (sample.get('prefill', 0) or 0) + weight * (
                    sample.get('decode', 0) or 0)
                source_domain = (domain.project(op.tables) if op.__type_id__ in
                                 {TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION} else domain)
                rate_key = oid
                if op.__type_id__ == TYPE_ID.SEM_JOIN and op.tables[1] in domain.tables:
                    rate_key = (oid, frozenset(domain.project((op.tables[1],)).rows))
                same_rows = (sample_mask is not None and
                             source_domain.tables == sample_mask.tables and
                             set(source_domain.surviving_rows()).issubset(set(sample_mask.rows)))
                if (not same_rows and work >= 5 and sample_work and actual_cost >= 500
                        and sample_cost):
                    actual_rate = actual_cost / work
                    sample_rate = self._executed_rates.get(rate_key, sample_cost / sample_work)
                    # A rate change on a small share of the step cannot reorder candidates, reprofiling it only adds latency.
                    material = abs(actual_rate - sample_rate) * work >= 0.1 * step_cost
                    if material and (actual_rate > 2 * sample_rate or
                                     sample_rate > 2 * actual_rate):
                        logger.info('Operator drift %s: %.0f versus %.0f token cost per evaluation',
                                    getattr(op, 'op_name', oid), sample_rate, actual_rate)
                        drift = True
                if work:
                    self._executed_rates[rate_key] = actual_cost / work
                if (not same_rows and op.__type_id__ == TYPE_ID.SEM_FILTER and
                        domain.tables == tuple(op.tables) and
                        sample_mask is not None and source_domain.count >= 5):
                    actual_sel = call['mask'].project(op.tables).count / source_domain.count
                    sample_sel = sample_mask.count / len(sample_mask.rows) if sample_mask.rows else 0
                    if abs(actual_sel - sample_sel) >= 0.5:
                        logger.info('Operator drift %s: selectivity %.2f versus %.2f',
                                    getattr(op, 'op_name', oid), sample_sel, actual_sel)
                        drift = True
                self._observations.setdefault(oid, deque(maxlen=64)).append({
                    'input_mask': domain, 'mask': call['mask'],
                    'work': call['evaluation_count'],
                    'prefill': call['prefill'], 'decode': call['decode'],
                })
        return drift

    def ingest_execution(self, group, subquery, op_oid_map, trace, context):
        """Use a completed full-batch pull as the next planning sample.

        Candidate plans must be costable on the same row domain. If the
        executed plan did not expose enough rows, retain the prior sample.
        """
        if trace is None:
            return False
        sampled_types = {TYPE_ID.SCAN, TYPE_ID.SOURCE, TYPE_ID.FILTER,
                         TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION,
                         TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}
        previous = {}
        samples = {}
        for op in subquery._get_ops(op_oid_map):
            if op.__type_id__ not in sampled_types:
                continue
            calls = trace.calls.get(op.oid, [])
            completed = [call for call in calls
                         if call['status'] == 'completed' and 'mask' in call
                         and not call['already_applied']]
            if not completed:
                group._execution_feedback_ready = False
                return False
            call = completed[-1]
            mask = call['mask']
            if op.__type_id__ not in {TYPE_ID.SCAN, TYPE_ID.SOURCE}:
                mask = mask.project(op.tables)
            work = call['evaluation_count'] if op.__type_id__ in {
                TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION, TYPE_ID.SEM_JOIN} else 0
            if op.__type_id__ in {TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION}:
                if work < mask.distinct_count(op.tables):
                    return False
            # A cached result still supplies its bitmap, but cannot replace a nonzero per-evaluation token sample with a zero-cost sample.
            old = getattr(op, '_profile_sample', None)
            token_sample = (call if work or old is None else old)
            samples[op.oid] = {
                'mask': mask, 'work': (work if work or old is None else old.get('work', 0)),
                'prefill': token_sample.get('prefill', 0),
                'decode': token_sample.get('decode', 0),
                'input_sz': call['input_rows'], 'output_sz': call['output_rows'],
                'sel': call['output_rows'] / call['input_rows'] if call['input_rows'] else 0,
                'sample_step': self.step_counter,
                'sample_query_step': trace.metadata.get('query_step'),
                'sample_origin': 'execution',
            }
            previous[op.oid] = old

        old_context = getattr(group, '_profile_cost_context', None)
        old_cost_context = getattr(self, '_cost_context', None)
        try:
            for oid, sample in samples.items():
                op_oid_map[oid]._profile_sample = sample
            self.cost_model.execution_context = context
            known = {}
            for oid, sample in samples.items():
                op = op_oid_map[oid]
                if op.__type_id__ in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER,
                                      TYPE_ID.SEM_PROJECTION}:
                    known[oid] = {}
            context['unary_known'] = known
            self._cost_context = context
            self._populate_subquery_group(group.subqueries, op_oid_map)
        except (KeyError, IndexError, ValueError):
            group._execution_feedback_ready = False
            for oid, sample in previous.items():
                op = op_oid_map[oid]
                if sample is None:
                    del op._profile_sample
                else:
                    op._profile_sample = sample
            self._cost_context = old_cost_context
            if old_context is not None:
                group._profile_cost_context = old_context
            return False
        group._profile_cost_context = context
        group._dump_sample_step = self.step_counter
        group._dump_sample_query_step = trace.metadata.get('query_step')
        group._last_complete_sample_step = self.step_counter
        group._execution_feedback_ready = True
        return True

    def _execution_inputs(self, subquery, op_oid_map):
        """Preview the next pull without advancing scans or changing join buffers."""
        leaves = subquery._get_leaf_ops()
        positions = {oid: op_oid_map[oid].cursor for oid in leaves}
        context = {'cursors': dict(positions), 'post_cursors': dict(positions),
                   'post_join_cursors': {}, 'leaves': {}, 'joins': {}, 'completed': {}}

        def subtree_leaves(oid):
            children = [c for c in subquery.ch_link.get(oid, []) if c != -1]
            return [oid] if not children else [leaf for child in children for leaf in subtree_leaves(child)]

        def buffered(batch, branch):
            cursors = {}
            batch = getattr(batch, '_raw_unfiltered', batch)
            retained = ProfileMask.from_batch(batch)
            for leaf in subtree_leaves(branch):
                source = op_oid_map[leaf]
                tables = ([source.table_name] if source.__type_id__ == TYPE_ID.SCAN else source.tables)
                if all(table in batch.tables for table in tables):
                    slot = batch.tables.index(tables[0]) if len(batch.idx) > 1 else 0
                    cursors[leaf] = batch.idx[slot] * source.batch_size
            return {'eof': batch.is_eof, 'cursors': cursors, 'buffer': retained,
                    'bindings': [(batch, branch)]}

        def peek(oid):
            op = op_oid_map[oid]
            children = [c for c in subquery.ch_link.get(oid, []) if c != -1]
            if not children:
                cursor = positions[oid]
                positions[oid] += op.batch_size
                # Scans and materialized sources do not advance on an EOF pull.
                if cursor < op.tot:
                    context['post_cursors'][oid] = cursor + op.batch_size
                return {'eof': cursor >= op.tot, 'cursors': {oid: cursor},
                        'buffer': None, 'bindings': []}
            if len(children) == 1:
                return peek(children[0])
            left_oid, right_oid = children
            phase = subquery._exec_state.get(oid, {}).get('phase', 'init')
            left = (peek(left_oid) if phase == 'init' or op.left_buffer is None
                    else buffered(op.left_buffer, left_oid))
            if left['eof']:
                return left
            if not op.right_eof:
                right = peek(right_oid)
                if right['eof']:
                    left = peek(left_oid)
                    right = (buffered(op.right_buffer[0], right_oid) if op.right_buffer
                             else right)
            else:
                index = op.right_cursor
                if index >= len(op.right_buffer) - 1:
                    left = peek(left_oid)
                    index = 0
                right = buffered(op.right_buffer[index], right_oid)
                # Replay advances independently of the join's output. Only record joins actually pulled; buffered child joins stay put.
                context['post_join_cursors'][oid] = index + 1
            context['joins'][oid] = (left['buffer'], right['buffer'])
            return {'eof': left['eof'] or right['eof'],
                    'cursors': {**left['cursors'], **right['cursors']}, 'buffer': None,
                    'bindings': left['bindings'] + right['bindings']}

        preview = peek(subquery.root)
        context['eof'] = preview['eof']
        context['cursors'].update(preview['cursors'])
        # Only retain buffers selected by the final preview. An empty old left buffer must not prune a new left batch fetched at the EOF transition.
        for batch, branch in preview['bindings']:
            retained = ProfileMask.from_batch(batch)
            for leaf in subtree_leaves(branch):
                source = op_oid_map[leaf]
                tables = ([source.table_name] if source.__type_id__ == TYPE_ID.SCAN else source.tables)
                if all(table in retained.tables for table in tables):
                    context['leaves'].setdefault(leaf, []).append(retained.project(tables))
            for oid in batch.fingerprint:
                tables = getattr(op_oid_map.get(oid), 'tables', [])
                if tables and all(table in retained.tables for table in tables):
                    context['completed'].setdefault(oid, []).append(retained)
        context['signature'] = tuple(sorted(context['cursors'].items()))
        return context

    def outer_batch_follows(self, subquery: SubQuery, op_oid_map: Dict[int, Operator]):
        context = self._execution_inputs(subquery, op_oid_map)
        if context['eof']:
            return False
        for oid, children in subquery.ch_link.items():
            op = op_oid_map[oid]
            if op.__type_id__ not in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}:
                continue
            if oid in context['post_join_cursors']:
                # Replaying: the cursor after this pull wraps on the next one.
                if context['post_join_cursors'][oid] >= len(op.right_buffer) - 1:
                    return True
            elif not op.right_eof:
                # Materializing: the right side reaches EOF on the next pull.
                leaves = [leaf for leaf in subquery._get_leaf_ops()
                          if self._descends(subquery, leaf, children[1])]
                if len(leaves) == 1:
                    source = op_oid_map[leaves[0]]
                    if (source.cursor < source.tot and
                            context['post_cursors'].get(leaves[0], 0) >= source.tot):
                        return True
        return False

    @staticmethod
    def _descends(subquery, oid, ancestor):
        while oid is not None:
            if oid == ancestor:
                return True
            oid = subquery.parent_link.get(oid)
        return False

    def _profile_subquery(self, subquery: SubQuery, op_oid_map: Dict[int, Operator]):
        subquery._use_llm(op_oid_map, self.model)
        operators = subquery._get_ops(op_oid_map)
        semantic = [op for op in operators if op.__type_id__ in {
            TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION, TYPE_ID.SEM_JOIN}]
        if len(semantic) > self.profile_budget:
            raise ValueError("Profile budget is smaller than the number of semantic operators")
        budget = _ProfileBudget(self.profile_budget)
        weights = [2 if op.__type_id__ == TYPE_ID.SEM_JOIN else 1 for op in semantic]
        spare = self.profile_budget - len(semantic)
        total_weight = sum(weights)
        allowances = [1 + spare * weight // total_weight for weight in weights] if weights else []
        if allowances:
            for i in range(spare - sum(allowance - 1 for allowance in allowances)):
                allowances[i % len(allowances)] += 1
        for op, allowance in zip(semantic, allowances):
            op.llm_client = _BudgetedModel(
                self.model, budget, allowance,
                (self.step_counter + 1) * 1000003 + op.oid * 1009)
        model_proxies = {op.oid: op.llm_client for op in semantic}
        execution_state = subquery._exec_state
        saved = []
        previous_samples = {op.oid: getattr(op, '_profile_sample', None) for op in operators}
        subquery._exec_state = {}
        try:
            for op in operators:
                attrs = {}
                if op.__type_id__ in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}:
                    names = ('left_buffer', 'right_buffer', 'right_cursor', 'right_eof', 'backup')
                    attrs.update((name, getattr(op, name)) for name in names)
                    op.left_buffer, op.right_buffer = None, []
                    op.right_cursor, op.right_eof, op.backup = 0, False, {}
                elif op.__type_id__ in {TYPE_ID.SCAN, TYPE_ID.SOURCE}:
                    attrs['cursor'] = op.cursor
                    op.cursor = self._sampling_context['cursors'].get(op.oid, op.cursor)
                if hasattr(op, 'active'):
                    attrs['active'] = op.active
                saved.append((op, attrs))

            for op in operators:
                if op.__type_id__ in {TYPE_ID.SCAN, TYPE_ID.SOURCE}:
                    op.process(is_profile=True)
            subquery.execute(op_oid_map, is_profile=True)
            for op in operators:
                sample = getattr(op, '_profile_sample', None)
                if sample is not None and sample is not previous_samples[op.oid]:
                    if op.oid in model_proxies:
                        sample['work'] = model_proxies[op.oid].used
                    sample['sample_step'] = self.step_counter + 1
                    sample['sample_origin'] = 'profile'
        finally:
            self.last_profile_evaluations = budget.used
            for op in semantic:
                op.llm_client = self.model
            subquery._exec_state = execution_state
            for op, attrs in saved:
                for name, value in attrs.items():
                    setattr(op, name, value)

    def _populate_subquery_group(self, subqueries: List[SubQuery], op_oid_map: Dict[int, Operator]):
        self.cost_model.call_overhead_cost = self.call_overhead_cost or 0.0
        if hasattr(self, '_cost_context'):
            self.cost_model.execution_context = self._cost_context
        else:
            live = getattr(self, '_live_subquery', subqueries[0])
            self.cost_model.execution_context = self._execution_inputs(live, op_oid_map)
        all_subq_prof = [0.0 for _ in subqueries]
        capture_candidates = tracing_enabled()
        self._candidate_breakdowns = []
        self._candidate_predictions = []
        self._last_prediction = None
        best_prediction_cost = float('inf')

        def get_cost_subq(subq: SubQuery):
            nonlocal best_prediction_cost
            # for each subquery, the estimation uses top-down dfs to accumulate cost for each operator based on its own profile and its children's output info.
            cost = {}
            details = {}
            def _dfs(oid: int):
                nonlocal cost
                if oid == -1:
                    return None

                child_infos = []
                for ch_oid in subq.ch_link.get(oid, [-1]):
                    if ch_oid == -1:
                        continue
                    child_infos.append(_dfs(ch_oid))

                op = op_oid_map[oid]
                op_cost, op_info = self.cost_model.get_cost_op(op, child_infos)
                # logger.info(f"Subq {subq.root} op {op.oid} cost: {op_cost}")
                cost[oid] = op_cost if op_cost is not None else 0.0
                details[oid] = {**op_info, 'amortization_factor': 1.0,
                                'amortized_prefill_cost': 0.0,
                                'amortized_decode_cost': 0.0,
                                'amortized_token_cost': 0.0}
                return op_info

            _dfs(subq.root)

            for join_oid in list(cost):
                join = op_oid_map[join_oid]
                if join.__type_id__ not in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}:
                    continue
                children = [oid for oid in subq.ch_link.get(join_oid, ()) if oid != -1]
                if len(children) != 2:
                    continue
                if getattr(join, 'right_eof', False):
                    factor = 0.0
                else:
                    leaf = children[0]
                    while subq.ch_link.get(leaf, [-1])[0] != -1:
                        leaf = subq.ch_link[leaf][0]
                    source = op_oid_map[leaf]
                    size = getattr(source, 'batch_size', 0)
                    outer = ceil(getattr(source, 'tot', 0) / size) if size else 1
                    factor = 1.0 / max(1, outer)

                leaf = children[1]
                while subq.ch_link.get(leaf, [-1])[0] != -1:
                    leaf = subq.ch_link[leaf][0]
                source = op_oid_map[leaf]
                size = getattr(source, 'batch_size', 0)
                inner = ceil(getattr(source, 'tot', 0) / size) if size else 1
                per_pull = 1.0 / max(1, inner)
                for child, child_factor in ((children[0], per_pull), (children[1], factor * per_pull)):
                    stack = [child]
                    while stack:
                        oid = stack.pop()
                        cost[oid] *= child_factor
                        details[oid]['amortization_factor'] *= child_factor
                        stack.extend(c for c in subq.ch_link.get(oid, ()) if c != -1)

            # Charge only the profiled source-batch pair. Future join pulls can use different batches and must not inherit this pair's cost rate.
            total = sum(cost.values())
            for join_oid in cost:
                join = op_oid_map[join_oid]
                if (join.__type_id__ not in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN} or
                        not getattr(join, 'right_eof', False)):
                    continue
                children = [oid for oid in subq.ch_link.get(join_oid, ()) if oid != -1]
                if len(children) != 2:
                    continue
                left_oid, right_oid = children
                while subq.ch_link.get(left_oid, [-1])[0] != -1:
                    left_oid = subq.ch_link[left_oid][0]
                while subq.ch_link.get(right_oid, [-1])[0] != -1:
                    right_oid = subq.ch_link[right_oid][0]
                right_source = op_oid_map[right_oid]
                current = self.cost_model.execution_context.get('cursors', {}).get(
                    right_oid, right_source.cursor) // right_source.batch_size
                future = [batch for batch in join.right_buffer
                          if not batch.is_eof and join.tables[1] in batch.tables
                          and batch.idx[batch.tables.index(join.tables[1])] > current]
                if not future:
                    continue
                def branch_filters(plan, root):
                    found = set()
                    stack = [root]
                    while stack:
                        oid = stack.pop()
                        if op_oid_map[oid].__type_id__ in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER}:
                            found.add(oid)
                        stack.extend(child for child in plan.ch_link.get(oid, ()) if child != -1)
                    return found
                incumbent = getattr(self, '_live_subquery', subq)
                incumbent_right = incumbent.ch_link[join_oid][1]
                if branch_filters(subq, right_oid) != branch_filters(incumbent, incumbent_right):
                    continue
                known_pair_cost = 0.0
                for batch in future:
                    batch_id = batch.idx[batch.tables.index(join.tables[1])]
                    rates = []
                    for observation in self._observations.get(join_oid, ()):
                        domain = observation['input_mask']
                        if (observation['work'] and join.tables[1] in domain.tables and
                                any(row[domain.tables.index(join.tables[1])][0] == batch_id
                                    for row in domain.rows)):
                            rates.append((observation['prefill'] + observation['decode']) /
                                         observation['work'])
                    if rates:
                        known_pair_cost += len(batch.data) * min(rates)
                if not known_pair_cost:
                    continue
                source = details[left_oid]['mask'].project([join.tables[0]])
                source_rows = set(source.surviving_rows())
                rejected = set()
                def path_to(oid):
                    if oid == join_oid:
                        return [oid]
                    for child in subq.ch_link.get(oid, ()):
                        if child != -1:
                            path = path_to(child)
                            if path:
                                return [oid] + path
                    return []
                for oid in path_to(subq.root)[:-1]:
                    op = op_oid_map[oid]
                    if (op.__type_id__ not in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER}
                            or tuple(op.tables) != (join.tables[0],)):
                        continue
                    sample_mask = (op._get_profile() or {}).get('mask')
                    if sample_mask is None or not source_rows.issubset(set(sample_mask.rows)):
                        continue
                    rejected.update(source_rows.difference(sample_mask.surviving_rows()))
                total += len(rejected) * known_pair_cost
            for oid, detail in details.items():
                factor = detail['amortization_factor']
                detail.update(amortized_prefill_cost=factor * detail['prefill_cost'],
                              amortized_decode_cost=factor * detail['decode_cost'],
                              amortized_token_cost=factor * detail['token_cost'])
            if 0 <= total < best_prediction_cost:
                best_prediction_cost = total
                self._last_prediction = details
            return total, details

        for i, subq in enumerate(subqueries):
            # logger.info(f"------- Profile populate: {i} --------\n")
            cost, details = get_cost_subq(subq)
            all_subq_prof[i] = cost
            self._candidate_predictions.append(details)
            if capture_candidates:
                self._candidate_breakdowns.append({
                    'candidate_id': i,
                    'total_cost': cost,
                    'plan': {'root_oid': subq.root, 'children': subq.ch_link},
                    'operators': {
                        str(oid): [entry['input_rows'], entry['output_sz'],
                                   entry['evaluation_count'], entry['token_cost'],
                                   entry['amortization_factor'],
                                   entry['amortized_token_cost']]
                        for oid, entry in details.items()
                    },
                })
            # logger.info(f'Subquery : {subq.verbose(oid_map=op_oid_map)}, estimated cost: {all_subq_prof[i]:.4f}')
        return all_subq_prof

    def _select_candidate(self, group, all_costs, op_oid_map):
        """Keep the incumbent when join cost evidence is weak."""
        valid = [i for i, cost in enumerate(all_costs) if cost >= 0]
        if not valid:
            return group.best_id, float('inf')
        minimum = min(all_costs[i] for i in valid)
        tied = [i for i in valid if isclose(all_costs[i], minimum,
                                            rel_tol=1e-9, abs_tol=1e-9)]
        if minimum == 0 and group.best_id in tied:
            # An empty replay pull gives no evidence for switching plans.
            return group.best_id, minimum
        has_join = any(op_oid_map[oid].__type_id__ in {TYPE_ID.JOIN, TYPE_ID.SEM_JOIN}
                       for oid in group.subqueries[0].ch_link)
        if has_join:
            best = min(valid, key=all_costs.__getitem__)
            margin = max(0.1 * minimum, (self.call_overhead_cost or 0.0) *
                         self._switch_amortization(group.best_id, best))
            near = [i for i in valid if all_costs[i] <= minimum + margin]
            logger.debug('Join plan margin %.1f: incumbent %d at %.1f, best %d at %.1f',
                         margin, group.best_id, all_costs[group.best_id], best, minimum)
            if group.best_id in near:
                return group.best_id, all_costs[group.best_id]
            selected = min(near, key=lambda i: (
                self._pushdown_score(group.subqueries[i]), all_costs[i], i))
            return selected, all_costs[selected]
        if len(tied) > 1:
            unbuffered = Profiler(self.model)
            unbuffered._cost_context = {
                'unary_known': {}, 'eof': self._cost_context.get('eof', False)}
            try:
                fresh_costs = unbuffered._populate_subquery_group(
                    [group.subqueries[i] for i in tied], op_oid_map)
            except (KeyError, IndexError, ValueError):
                logger.debug('Full sample domain unavailable for candidate tie', exc_info=True)
            else:
                fresh_minimum = min(fresh_costs)
                logger.info('Live-cost tie %s at %.4f; uncached sample costs %s',
                            tied, minimum, dict(zip(tied, fresh_costs)))
                tied = [i for i, cost in zip(tied, fresh_costs)
                        if isclose(cost, fresh_minimum, rel_tol=1e-9, abs_tol=1e-9)]
        selected = group.best_id if group.best_id in tied else tied[0]
        return selected, all_costs[selected]

    def _switch_amortization(self, incumbent, best):
        """Amortization of the operators whose cost a switch changes.

        Inner-side costs are spread over the outer batches, so the round trip
        a switch must save is spread the same way.
        """
        predictions = self._candidate_predictions
        if not (0 <= incumbent < len(predictions) and 0 <= best < len(predictions)):
            return 1.0
        old, new = predictions[incumbent], predictions[best]
        factors = [
            entry['amortization_factor']
            for oid in set(old) | set(new)
            if not isclose(old.get(oid, {}).get('amortized_token_cost', 0.0),
                           new.get(oid, {}).get('amortized_token_cost', 0.0),
                           rel_tol=1e-9, abs_tol=1e-9)
            # An operator left without work, such as one above an empty join costs no round trip in that plan.
            for entry in (old.get(oid), new.get(oid))
            if entry and entry['amortized_token_cost'] > 0
        ]
        return max(factors, default=1.0)

    @staticmethod
    def _pushdown_score(subq):
        """Lower when operators sit closer to the leaves."""
        def _calculate_pushdown_score(cur_node: int):
            if cur_node == -1:
                return 0, 0
            children = subq.ch_link.get(cur_node, [-1, -1])
            if children == [-1, -1]:
                return 1, 1

            total_leaves = 0
            total_cost = 0

            for child in children:
                child_leaves, child_cost = _calculate_pushdown_score(child)
                total_leaves += child_leaves
                total_cost += child_cost
            return total_leaves, total_cost + total_leaves

        return _calculate_pushdown_score(subq.root)[1]

    def _get_profile_subq(self, candidate_subqueries: SubQueryGroup):
        min_score = float('inf')
        min_id = -1
        for i, subq in enumerate(candidate_subqueries.subqueries):
            score = self._pushdown_score(subq)
            if score < min_score:
                min_score = score
                min_id = i
        # logger.info(f"Selected subquery id for profiling: {min_id}")
        return candidate_subqueries.subqueries[min_id]

    @dump_profile
    def profile(self, candidate_subqueries: SubQueryGroup, op_oid_map: Dict[int, Operator], profile_enabled: bool = True):
        self._op_map = op_oid_map
        self._live_subquery = candidate_subqueries.subqueries[candidate_subqueries.best_id]
        self._sampling_context = self._execution_inputs(self._live_subquery, op_oid_map)
        profile_subq = self._get_profile_subq(candidate_subqueries)
        # logger.info(f"Profiling subquery: {profile_subq.verbose(oid_map=op_oid_map)}")
        profile_subq._profile_enabled = profile_enabled
        rollback = False
        needs_sample = not hasattr(candidate_subqueries, '_profile_cost_context')
        sample_age = (self.step_counter + 1
                      - getattr(candidate_subqueries, '_last_complete_sample_step',
                                self.step_counter + 1))
        if (needs_sample or getattr(candidate_subqueries, '_needs_reprofile', False)
                or sample_age >= PROFILE_MAX_SAMPLE_AGE
                or (self.step_counter % PROFILE_REFRESH_INTERVAL == 0
                    and not getattr(candidate_subqueries, '_execution_feedback_ready', False))):
            candidate_subqueries._dump_sample_step = self.step_counter + 1
            self._profile_subquery(profile_subq, op_oid_map)
            candidate_subqueries._last_complete_sample_step = self.step_counter + 1
            candidate_subqueries._needs_reprofile = False
            candidate_subqueries._feedback_stats = {}
            context = self._sampling_context
            self.cost_model.execution_context = context
            known = {}
            for op in profile_subq._get_ops(op_oid_map):
                if op.__type_id__ in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION}:
                    sample = getattr(op, '_profile_sample', None)
                    if sample is not None:
                        _, known[op.oid] = self.cost_model._unary_cache(op, sample['mask'])
            context['unary_known'] = known
            candidate_subqueries._profile_cost_context = context
            # Profiling restores execution state before candidate costing.
        self._cost_context = candidate_subqueries._profile_cost_context
        all_costs = self._populate_subquery_group(candidate_subqueries.subqueries, op_oid_map)
        candidate_subqueries.best_id, min_cost = self._select_candidate(
            candidate_subqueries, all_costs, op_oid_map)
        self._last_prediction = self._candidate_predictions[candidate_subqueries.best_id]
        candidate_subqueries._predicted_step_cost = sum(
            detail['token_cost'] for detail in self._last_prediction.values())
        candidate_subqueries._predicted_cost_supported = all(
            detail['cost_model_supported'] for detail in self._last_prediction.values())

        for cost, subq in zip(all_costs, candidate_subqueries.subqueries):
            logger.info(f"Subquery : {subq.verbose(oid_map=op_oid_map)}, estimated cost: {cost:.4f}")

        self.step_counter += 1
        return candidate_subqueries.best_id, min_cost, rollback


_profiler: Optional[Profiler] = None

def init_profiler(model, profile_budget: int = 20) -> Profiler:
    global _profiler
    _profiler = Profiler(model, profile_budget=profile_budget)
    return _profiler

def get_profiler() -> Profiler:
    if _profiler is None:
        raise RuntimeError("Profiler has not been initialised. Call init_profiler(model) first.")
    return _profiler
