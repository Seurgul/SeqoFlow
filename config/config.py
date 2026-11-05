from pathlib import Path
from typing import Any, Optional, Tuple
import functools
import json
import math
import random
import re
import pandas as pd
import logging
from config.randomize import (FLUCTUATION_PERCENT, FLUCTUATION_MODE,
                              FLUCTUATION_BURSTS,
                              REGIME_SEGMENTS, fluctuate_op_config,
                              regime_op_configs, crossover_op_configs,
                              CROSSOVER_SOURCES,
                              get_batch_size, get_config_seed, get_use_config)

HOME = Path(__file__).resolve().parent.parent

# plan_idx -> original config payload
_ACTIVE_PLAN_CONFIGS = {}
_BATCH_PLAN_CONFIGS = {}
_SOURCE_BATCH_COUNTS = {}

logger = logging.getLogger(__name__)

def _load_static_plan_config(dataset, plan_idx):
    with open(HOME / "config" / dataset / f"t{plan_idx}_config.json", "r") as f:
        return json.load(f)


def set_active_plan_config(dataset, plan_idx, configs):
    key = f"{dataset}_{int(plan_idx)}"
    _ACTIVE_PLAN_CONFIGS[key] = configs
    _BATCH_PLAN_CONFIGS.pop(key, None)
    _SOURCE_BATCH_COUNTS.pop(key, None)


def clear_active_plan_config(dataset, plan_idx):
    """Remove all runtime configuration state for one plan."""
    key = f"{dataset}_{int(plan_idx)}"
    _ACTIVE_PLAN_CONFIGS.pop(key, None)
    _BATCH_PLAN_CONFIGS.pop(key, None)
    _SOURCE_BATCH_COUNTS.pop(key, None)


def _batch_count(scan_args):
    """Source batches implied by a scan config's row range."""
    return math.ceil(max(0, scan_args['rows'][1] - scan_args['rows'][0]) / get_batch_size())


def register_scan_size(dataset, plan_idx, table, row_count):
    """Record actual scan size before runtime burst generation."""
    key = f"{dataset}_{int(plan_idx)}"
    counts = _SOURCE_BATCH_COUNTS.setdefault(key, {})
    count = math.ceil(row_count / get_batch_size())
    if counts.get(table) != count:
        counts[table] = count
        _BATCH_PLAN_CONFIGS.pop(key, None)


def get_active_plan_config(dataset, plan_idx):
    return _ACTIVE_PLAN_CONFIGS.get(f"{dataset}_{int(plan_idx)}")


def prepare_plan_config(dataset, plan_idx):
    # Only register the baseline here. Bursts are generated on first runtime use.
    configs = _load_static_plan_config(dataset, plan_idx)
    set_active_plan_config(dataset, plan_idx, configs)
    return configs


def prepare_scan_config(dataset, plan_idx):
    """Load only scan row ranges from the static plan configuration."""
    scans = [config for config in _load_static_plan_config(dataset, plan_idx)
             if config['op_type'] == 'scan']
    set_active_plan_config(dataset, plan_idx, scans)
    return scans


def _find_op_conf(configs, op_name):
    conf = None
    if isinstance(configs, list):
        conf = next((c for c in configs if isinstance(c, dict) and c['op_args']['op_name'] == op_name), None)
    elif isinstance(configs, dict):
        if "configs" in configs and isinstance(configs["configs"], list):
            conf = next((c for c in configs["configs"] if isinstance(c, dict) and c['op_args']['op_name'] == op_name), None)
        else:
            conf = configs.get(str(op_name), configs.get(op_name, None))
    return conf


def _random_bursts(count, affected, rng):
    """One random contiguous source-batch window with exact requested coverage."""
    if not 0 <= affected <= count:
        raise ValueError('Affected source batches must be in [0, count]')
    if not affected:
        return []
    start = rng.randrange(count - affected + 1)
    return [(start, start + affected)]


def _crossover_bursts(count, affected, bursts, rng):
    """Up to `bursts` separate random windows covering exactly `affected` batches."""
    if not 0 <= affected <= count:
        raise ValueError('Affected source batches must be in [0, count]')
    if not affected:
        return []
    # Separate windows need a gap between each pair.
    k = max(1, min(bursts, affected, count - affected + 1))
    cuts = sorted(rng.sample(range(1, affected), k - 1))
    lengths = [b - a for a, b in zip([0] + cuts, cuts + [affected])]
    spare = count - affected - (k - 1)
    bars = sorted(rng.sample(range(spare + k), k))
    gaps = [b - a - 1 for a, b in zip([-1] + bars, bars + [spare + k])]
    ranges, start = [], 0
    for index, length in enumerate(lengths):
        start += gaps[index] + (1 if index else 0)
        ranges.append((start, start + length))
        start += length
    return ranges


def _regime_schedule(count, affected, first, second, phase):
    """Alternate A/B blocks with exactly the requested B-batch coverage."""
    if not 0 <= affected <= count:
        raise ValueError('Affected source batches must be in [0, count]')
    if affected in (0, count):
        return [second if affected else first] * count
    segments = min(REGIME_SEGMENTS, count, 2 * min(affected, count - affected))
    segments -= segments % 2
    per_regime = segments // 2
    lengths = {
        0: [((count - affected) // per_regime) + (index < (count - affected) % per_regime)
            for index in range(per_regime)],
        1: [(affected // per_regime) + (index < affected % per_regime)
            for index in range(per_regime)],
    }
    schedule = []
    for segment in range(segments):
        regime = (segment + phase) % 2
        schedule.extend([second if regime else first] * lengths[regime][segment // 2])
    return schedule


def _prepare_batch_configs(dataset, plan_idx, configs):
    """Assign frozen fluctuation snapshots to each source's batches."""
    if FLUCTUATION_MODE not in {'legacy', 'regime', 'crossover', 'burst'}:
        raise ValueError(f'Unknown fluctuation mode: {FLUCTUATION_MODE}')
    if not 0 <= FLUCTUATION_PERCENT <= 100:
        raise ValueError('FLUCTUATION_PERCENT must be between 0 and 100')
    key = f'{dataset}_{int(plan_idx)}'
    schedules = {}
    if FLUCTUATION_MODE == 'crossover':
        first, second = crossover_op_configs(configs, dataset, int(plan_idx))
        for entry in configs:
            if entry['op_type'] != 'scan':
                continue
            args = entry['op_args']
            table = args['table_name']
            count = _SOURCE_BATCH_COUNTS.get(key, {}).get(table,
                _batch_count(args))
            boundary = count // 4
            schedules[table] = ([first] * boundary + [second] * (count - 2 * boundary)
                                + [first] * boundary)
            logger.info('Crossover %s t%s source=%s: A[%d] B[%d] A[%d]',
                        dataset, plan_idx, table, boundary,
                        count - 2 * boundary, boundary)
        _BATCH_PLAN_CONFIGS[key] = schedules
        return
    if FLUCTUATION_MODE == 'burst':
        # The crossover phases, with B in seeded random bursts that cover the
        # requested percentage of each source's batches instead of the middle.
        first, second = crossover_op_configs(configs, dataset, int(plan_idx))
        for entry in configs:
            if entry['op_type'] != 'scan':
                continue
            args = entry['op_args']
            table = args['table_name']
            count = _SOURCE_BATCH_COUNTS.get(key, {}).get(table,
                _batch_count(args))
            affected = round(count * FLUCTUATION_PERCENT / 100)
            rng = random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{table}:crossover-burst')
            ranges = _crossover_bursts(count, affected, FLUCTUATION_BURSTS, rng)
            schedules[table] = [first] * count
            for start, end in ranges:
                schedules[table][start:end] = [second] * (end - start)
            logger.info('Crossover %s t%s source=%s: bursts B=%s of %d batches, '
                        'percent=%g max_bursts=%d', dataset, plan_idx, table, ranges,
                        count, FLUCTUATION_PERCENT, FLUCTUATION_BURSTS)
        _BATCH_PLAN_CONFIGS[key] = schedules
        return
    if FLUCTUATION_MODE == 'regime':
        first, second, cohorts = regime_op_configs(configs, dataset, plan_idx)
        for entry in configs:
            if entry['op_type'] != 'scan':
                continue
            args = entry['op_args']
            table = args['table_name']
            count = _SOURCE_BATCH_COUNTS.get(key, {}).get(table,
                _batch_count(args))
            phase = random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{table}:phase').randrange(2)
            affected = round(count * FLUCTUATION_PERCENT / 100)
            schedules[table] = _regime_schedule(count, affected, first, second, phase)
            transitions = sum(left is not right for left, right in
                              zip(schedules[table], schedules[table][1:]))
            logger.info('Regimes %s t%s source=%s batches=%d affected=%d percent=%g '
                        'segments=%d phase=%d cohorts=%s',
                        dataset, plan_idx, table, count, affected, FLUCTUATION_PERCENT,
                        transitions + 1 if count else 0, phase, cohorts)
        _BATCH_PLAN_CONFIGS[key] = schedules
        return
    if FLUCTUATION_PERCENT:
        changed = fluctuate_op_config(configs)
        logger.info('Fluctuation %s t%s evaluated=%d configured operators; changed=%d',
                    dataset, plan_idx, len(configs),
                    sum(before != after for before, after in zip(configs, changed)))
        for baseline, fluctuation in zip(configs, changed):
            if baseline != fluctuation:
                logger.info('Fluctuation %s t%s %s: %s -> %s', dataset, plan_idx,
                            baseline['op_args']['op_name'], baseline['op_args'],
                            fluctuation['op_args'])
        for entry in configs:
            if entry['op_type'] != 'scan':
                continue
            args = entry['op_args']
            table = args['table_name']
            count = _SOURCE_BATCH_COUNTS.get(key, {}).get(table,
                _batch_count(args))
            rng = random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{table}:burst')
            ranges = _random_bursts(count, math.ceil(count*FLUCTUATION_PERCENT/100), rng)
            batches = [configs]*count
            for start, end in ranges:
                batches[start:end] = [changed]*(end-start)
            schedules[table] = batches
            logger.info('Fluctuation %s t%s source=%s burst=%s percent=%s',
                        dataset, plan_idx, table, ranges, FLUCTUATION_PERCENT)
    _BATCH_PLAN_CONFIGS[key] = schedules


def get_batch_plan_config(dataset, plan_idx, batch, source_tables=None):
    """Get the config assigned to the operator's source batch."""
    base = get_active_plan_config(dataset, plan_idx)
    if base is None or not get_use_config() or (FLUCTUATION_MODE == 'legacy'
                                                and FLUCTUATION_PERCENT == 0):
        return base
    if batch is None or not getattr(batch, 'tables', None):
        return base
    if getattr(batch, 'is_eof', False) or all(i < 0 for i in batch.idx):
        return base
    if f'{dataset}_{int(plan_idx)}' not in _BATCH_PLAN_CONFIGS:
        _prepare_batch_configs(dataset, plan_idx, base)
    batches = _BATCH_PLAN_CONFIGS.get(f'{dataset}_{int(plan_idx)}')
    if not batches or batch is None or not getattr(batch, 'tables', None):
        return base
    tables = [t for t in (source_tables or batch.tables) if t in batch.tables]
    for table in tables:
        index = batch.tables.index(table)
        if index < len(batch.idx) and batch.idx[index] >= 0:
            table_configs = batches.get(table, [])
            if batch.idx[index] < len(table_configs):
                return table_configs[batch.idx[index]]
    return base


def get_join_batch_config(dataset, plan_idx, *batches):
    """Resolve a pair's settings from stable source-batch identities."""
    base = get_active_plan_config(dataset, plan_idx)
    if FLUCTUATION_MODE in {'regime', 'crossover', 'burst'}:
        tables = sorted({table for batch in batches if batch is not None
                         for table in (getattr(batch, 'tables', ()) or ())})
        if not tables:
            return base
        canonical = (CROSSOVER_SOURCES[int(plan_idx)]
                     if FLUCTUATION_MODE in {'crossover', 'burst'} else tables[0])
        for batch in batches:
            if batch is not None and canonical in (getattr(batch, 'tables', ()) or ()):
                return get_batch_plan_config(dataset, plan_idx, batch, [canonical])
        return base
    for batch in batches:
        if batch is None:
            continue
        for table in getattr(batch, 'tables', ()) or ():
            config = get_batch_plan_config(dataset, plan_idx, batch, [table])
            if config is not base:
                return config
    return base


def configured_source_rows(dataset, plan_idx, op_name, table, batch_id,
                           population, selectivity):
    """Pick source identities independently of operator placement and input size."""
    rng = random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{op_name}:{table}:{batch_id}')
    count = max(0, min(population, int(selectivity * population)))
    return set(rng.sample(range(population), count))


def config_input_len(data, key, tlen):
    if data is None or len(data) == 0 or key not in data.columns:
        return data
    if not (data[key].dtype == object or data[key].dtype == 'str'):
        data[key] = data[key].astype(str)

    # if data[key] is not enough, repeat the string until it reaches the desired length. else, truncate it.
    def adjust_string(s):
        if s is None:
            return s
        s = str(s)
        if len(s) < tlen:
            repeat_times = (tlen + len(s) - 1) // len(s)
            return (s * repeat_times)[:tlen]
        else:
            return s[:tlen]
    data[key] = data[key].apply(adjust_string)
    return data

def truncate_join_input(data, key, tlen=None):
    """Truncate a join field to its configured length, capped at 2,000 characters."""
    if data is None or len(data) == 0 or key not in data.columns:
        return data
    limit = min(tlen, 2000) if isinstance(tlen, int) and tlen > 0 else 2000
    data[key] = data[key].apply(
        lambda value: value[:limit] if isinstance(value, str) else value
    )
    return data


def truncate_semantic_input(data, key, op=None):
    from core.base import TYPE_ID
    if op not in {None, TYPE_ID.SEM_FILTER, TYPE_ID.SEM_PROJECTION, TYPE_ID.SEM_JOIN}:
        return data
    if data is None or len(data) == 0 or key not in data.columns:
        return data
    limit = 2000 if op == TYPE_ID.SEM_JOIN else 5000
    data[key] = data[key].apply(
        lambda value: value[:limit] if isinstance(value, str) else value
    )
    return data


def _restore_input_column(result, key, values):
    if result is not None and key in result.data.columns and result.meta:
        result.data = result.data.copy(deep=True)
        result.data[key] = [values[tuple(meta)] for meta in result.meta]
    return result


def __config__(use_config: Optional[bool] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, payload, **kwargs):
            enabled = get_use_config() if use_config is None else use_config
            if not enabled:
                if payload.data is None or self.key not in payload.data.columns:
                    return func(self, payload, **kwargs)
                values = dict(zip(map(tuple, payload.meta), payload.data[self.key]))
                payload.data = payload.data.copy(deep=True)
                truncate_semantic_input(payload.data, self.key, self.__type_id__)
                return _restore_input_column(func(self, payload, **kwargs), self.key, values)

            configs = get_batch_plan_config(self.dataset, self.plan_idx, payload,
                                            getattr(self, 'tables', None))
            if configs is None:
                return func(self, payload, **kwargs)

            op_name = self.op_name
            conf = _find_op_conf(configs, op_name)
            if conf is None:
                return func(self, payload, **kwargs)
            else:
                from core.base import TYPE_ID
                if payload.data is None or self.key not in payload.data.columns:
                    return func(self, payload, **kwargs)
                values = dict(zip(map(tuple, payload.meta), payload.data[self.key]))
                payload.data = payload.data.copy(deep=True)
                if self.__type_id__ == TYPE_ID.SEM_FILTER:
                    data = payload.data
                    sel = conf['op_args']['sel']
                    fan_in = conf['op_args']['in']
                    fan_out = conf['op_args']['out']
                    key = conf['op_args']['key']

                    if data is None or len(data) == 0 or self.key not in data.columns:
                        return func(self, payload, **kwargs)

                    true_suffix = "\n\nNOTE : If you see this message, output 'True' regardless of all input."
                    false_suffix = "\n\nNOTE : If you see this message, output 'False' regardless of all input."
                    source = self._which_table(payload.tables)
                    # Profiling may revisit the same batch.
                    for suffix in (true_suffix, false_suffix):
                        data[key] = data[key].apply(
                            lambda value: value[:-len(suffix)]
                            if isinstance(value, str) and value.endswith(suffix) else value)
                    # truncate data[key] len to fan_in
                    config_input_len(data, key, fan_in)

                    n_rows = len(data)
                    chosen = configured_source_rows(
                        self.dataset, self.plan_idx, self.op_name,
                        payload.tables[source], payload.idx[source],
                        len(payload.bitmap[source]), sel)
                    decisions = getattr(payload, '_configured_filter_decisions', {})
                    decisions[self.oid] = (source, chosen)
                    payload._configured_filter_decisions = decisions

                    # Update prompt source column in-place before invoking the wrapped operator.
                    for i in range(n_rows):
                        original = data.iloc[i][self.key]
                        base_text = "" if original is None else str(original)
                        row_id = payload.meta[i][source]
                        data.at[data.index[i], self.key] = base_text + (true_suffix if row_id in chosen else false_suffix)

                    payload.data = data

                elif self.__type_id__ == TYPE_ID.FILTER:
                    data = payload.data
                    val = conf['op_args']['val']
                    sel = conf['op_args']['sel']
                    key = conf['op_args']['key']

                    if data is None or len(data) == 0 or self.key not in data.columns:
                        return func(self, payload, **kwargs)

                    n_rows = len(data)
                    source = self._which_table(payload.tables)
                    chosen = configured_source_rows(
                        self.dataset, self.plan_idx, self.op_name,
                        payload.tables[source], payload.idx[source],
                        len(payload.bitmap[source]), sel)
                    decisions = getattr(payload, '_configured_filter_decisions', {})
                    decisions[self.oid] = (source, chosen)
                    payload._configured_filter_decisions = decisions

                    for i in range(n_rows):
                        if payload.meta[i][source] in chosen:
                            data.at[data.index[i], self.key] = val
                        elif isinstance(val, str) and val in str(data.at[data.index[i], self.key]):
                            data.at[data.index[i], self.key] = '[CONFIG_FALSE]'

                    payload.data = data

                elif self.__type_id__ == TYPE_ID.SEM_PROJECTION:
                    data = payload.data
                    key = conf['op_args']['key']
                    fan_in = conf['op_args']['in']
                    fan_out = conf['op_args']['out']

                    if data is None or len(data) == 0 or self.key not in data.columns:
                        return func(self, payload, **kwargs)

                    config_input_len(data, key, fan_in)
                    # config output len
                    self.llm_args['max_tokens'] = fan_out

                    payload.data = data

                return _restore_input_column(func(self, payload, **kwargs), self.key, values)
        return wrapper
    return decorator


def __config_scan__(use_config: Optional[bool] = None):
    """Apply configured scan row ranges regardless of the randomization mode."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            plan_idx = kwargs.get('plan_idx')
            op_name = kwargs.get('op_name')
            dataset = kwargs.get('dataset')
            if plan_idx is None or op_name is None or dataset is None:
                return func(self, *args, **kwargs)
            configs = get_active_plan_config(dataset, plan_idx)
            if configs is None:
                return func(self, *args, **kwargs)

            conf = _find_op_conf(configs, op_name)
            if conf is None:
                return func(self, *args, **kwargs)

            func(self, *args, **kwargs)

            row = conf['op_args']['rows']
            self.table = pd.concat([self.table for _ in range(2)], ignore_index=True)  # Expand the table to ensure the row slice is valid
            self.table = self.table.iloc[row[0]:row[1]]
            self.table.reset_index(drop=True, inplace=True)
            self.tot = len(self.table)
            register_scan_size(dataset, plan_idx, self.table_name, self.tot)
            return
        return wrapper
    return decorator


def __config_join__(use_config: Optional[bool] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            enabled = get_use_config() if use_config is None else use_config
            kwargs['truncate_input'] = truncate_semantic_input
            if not enabled:
                kwargs['use_config'] = False
                return func(self, *args, **kwargs)

            kwargs['use_config'] = True
            configs = get_active_plan_config(self.dataset, self.plan_idx)
            if configs is None:
                kwargs['use_config'] = False
                return func(self, *args, **kwargs)

            op_name = self.op_name
            payload = args[0] if args else kwargs.get('payload')
            side = args[1] if len(args) > 1 else kwargs.get('side')
            if side == 'right' and payload is not None and not payload.is_eof:
                right = (self.right_buffer[self.right_cursor]
                         if self.right_eof else payload)
                configs = get_join_batch_config(self.dataset, self.plan_idx,
                                                self.left_buffer, right)
            conf = _find_op_conf(configs, op_name)
            if conf is None:
                kwargs['use_config'] = False
                return func(self, *args, **kwargs)

            kwargs['left_sel'] = conf['op_args'].get('left_sel', None)
            kwargs['right_sel'] = conf['op_args'].get('right_sel', None)

            kwargs['left_in'] = conf['op_args'].get('left_in', None)
            kwargs['right_in'] = conf['op_args'].get('right_in', None)
            kwargs['left_on'] = conf['op_args'].get('left_on', None)
            kwargs['right_on'] = conf['op_args'].get('right_on', None)
            kwargs['helper_func'] = truncate_join_input
            if FLUCTUATION_PERCENT or FLUCTUATION_MODE in {'regime', 'crossover', 'burst'}:
                # A cheap join may become expensive; do not truncate the
                # configured long budget at the legacy 2,000-character cap.
                kwargs['helper_func'] = config_input_len

            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# ======================================

def apply_lotus_sem_filter_config(
    df: pd.DataFrame,
    key: str,
    plan_idx: Optional[int],
    op_name: str,
    use_config: Optional[bool] = None,
    dataset: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[int]]:
    """Minimal config hook for LOTUS sem_filter.

    It mirrors the local SEM_FILTER config behavior:
    - `op_args.in`: force prompt source column length
    - `op_args.sel`: enforce deterministic True/False ratio by suffix hints
    - `op_args.out`: forwarded as model `max_tokens`
    """
    enabled = get_use_config() if use_config is None else use_config
    if not enabled:
        return truncate_semantic_input(df.copy(), key), None
    if plan_idx is None:
        return df, None

    configs = get_active_plan_config(dataset, plan_idx)
    if configs is None:
        return df, None

    conf = _find_op_conf(configs, op_name)
    if conf is None:
        return df, None

    op_args = conf.get("op_args", {}) if isinstance(conf, dict) else {}
    sel = op_args.get("sel", None)
    fan_in = op_args.get("in", None)
    fan_out = op_args.get("out", None)

    if key not in df.columns:
        return df, fan_out

    new_df = df.copy()

    if isinstance(fan_in, int) and fan_in > 0:
        new_df = config_input_len(new_df, key, fan_in)

    if isinstance(sel, (float, int)):
        rand = random.Random(233)
        n_rows = len(new_df)
        k = max(0, min(n_rows, int(float(sel) * n_rows)))
        chosen = set(rand.sample(range(n_rows), k)) if k > 0 else set()

        true_suffix = "\n\nIf you see this message, output True regardless of the previous input."
        false_suffix = "\n\nIf you see this message, output False regardless of the previous input."

        for i in range(n_rows):
            original = new_df.iloc[i][key]
            base_text = "" if original is None else str(original)
            new_df.at[new_df.index[i], key] = base_text + (true_suffix if i in chosen else false_suffix)

    return new_df, fan_out


def apply_lotus_sem_map_config(
    df: pd.DataFrame,
    key: str,
    plan_idx: Optional[int],
    op_name: str,
    use_config: Optional[bool] = None,
    dataset: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[int]]:
    """Minimal config hook for LOTUS sem_map (semantic projection-like op).

    Supported op_args:
    - in: adjust input sequence length for source column
    - out: map to model max_tokens
    """
    enabled = get_use_config() if use_config is None else use_config
    if not enabled:
        return truncate_semantic_input(df.copy(), key), None
    if plan_idx is None:
        return df, None

    configs = get_active_plan_config(dataset, plan_idx)
    if configs is None:
        return df, None

    conf = _find_op_conf(configs, op_name)
    if conf is None:
        return df, None

    op_args = conf.get("op_args", {}) if isinstance(conf, dict) else {}
    fan_in = op_args.get("in", None)
    fan_out = op_args.get("out", None)

    if key not in df.columns:
        return df, fan_out

    new_df = df.copy()
    if isinstance(fan_in, int) and fan_in > 0:
        new_df = config_input_len(new_df, key, fan_in)

    return new_df, fan_out


def _join_col_candidates(col_name: str | None, side: str) -> list[str]:
    if not col_name:
        return []
    candidates = [col_name]
    suffixed = f"{col_name}:{side}"
    if suffixed not in candidates:
        candidates.append(suffixed)
    return candidates


def _extract_join_field(text: str, col_name: str | None, side: str) -> tuple[str | None, str | None]:
    for candidate in _join_col_candidates(col_name, side):
        label = candidate.capitalize()
        pattern = rf"\[{re.escape(label)}\]:\s*«(.*?)»"
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1), label
    return None, None


def _replace_join_field_len(text: str, label: str, tlen: int) -> str:
    pattern = rf"(\[{re.escape(label)}\]:\s*«)(.*?)(»)"

    def adjust_value(value: str) -> str:
        return value[:min(tlen, 2000)]

    def repl(match: re.Match[str]) -> str:
        return f"{match.group(1)}{adjust_value(match.group(2))}{match.group(3)}"

    return re.sub(pattern, repl, text, count=1, flags=re.DOTALL)


def apply_lotus_sem_join_docs_config(
    docs: list[dict[str, Any]],
    plan_idx: Optional[int],
    op_name: str,
    use_config: Optional[bool] = None,
    dataset: Optional[str] = None,
) -> Tuple[list[dict[str, Any]], Optional[int]]:
    """Minimal config hook for LOTUS sem_join.

    Supported op_args:
    - left_on/right_on: source columns used in join prompt text
    - left_in/right_in: adjust input length of each join side independently
    - left_sel/right_sel: enforce deterministic two-sided selectivity by suffix hints
    - out: forwarded as model max_tokens
    """
    enabled = get_use_config() if use_config is None else use_config
    if not enabled:
        return [dict(doc, text=str(doc.get("text", ""))[:5000]) for doc in docs], None
    if plan_idx is None:
        return docs, None

    configs = get_active_plan_config(dataset, plan_idx)
    if configs is None:
        return docs, None

    conf = _find_op_conf(configs, op_name)
    if conf is None:
        return docs, None

    op_args = conf.get("op_args", {}) if isinstance(conf, dict) else {}
    left_on = op_args.get("left_on", None)
    right_on = op_args.get("right_on", None)
    left_sel = op_args.get("left_sel", None)
    right_sel = op_args.get("right_sel", None)
    left_in = op_args.get("left_in", None)
    right_in = op_args.get("right_in", None)
    fan_out = op_args.get("out", None)

    if not docs:
        return docs, fan_out

    new_docs = [{"text": str(doc.get("text", "")), "image": dict(doc.get("image", {}))} for doc in docs]

    parsed_fields: list[dict[str, Any]] = []
    left_values: list[str] = []
    right_values: list[str] = []

    for doc in new_docs:
        text = doc["text"]
        left_value, left_label = _extract_join_field(text, left_on, "left")
        right_value, right_label = _extract_join_field(text, right_on, "right")
        parsed_fields.append(
            {
                "left_value": left_value,
                "left_label": left_label,
                "right_value": right_value,
                "right_label": right_label,
            }
        )
        # print(left_label, left_value)
        # print(right_label, right_value)
        if left_value is not None:
            left_values.append(left_value)
        if right_value is not None:
            right_values.append(right_value)

    # Apply per-side input length controls.
    if isinstance(left_in, int) and left_in > 0:
        for i, doc in enumerate(new_docs):
            left_label = parsed_fields[i]["left_label"]
            if left_label is not None:
                doc["text"] = _replace_join_field_len(doc["text"], left_label, left_in)
    if isinstance(right_in, int) and right_in > 0:
        for i, doc in enumerate(new_docs):
            right_label = parsed_fields[i]["right_label"]
            if right_label is not None:
                doc["text"] = _replace_join_field_len(doc["text"], right_label, right_in)

    # Apply deterministic two-sided selectivity controls.
    apply_left_sel = isinstance(left_sel, (float, int))
    apply_right_sel = isinstance(right_sel, (float, int))

    chosen_left_values: set[str] = set()
    chosen_right_values: set[str] = set()
    rand = random.Random(233)

    if apply_left_sel and left_values:
        unique_left = list(dict.fromkeys(left_values))
        k_left = max(0, min(len(unique_left), int(float(left_sel) * len(unique_left))))
        chosen_left_values = {unique_left[i] for i in rand.sample(range(len(unique_left)), k_left)} if k_left > 0 else set()

    if apply_right_sel and right_values:
        unique_right = list(dict.fromkeys(right_values))
        k_right = max(0, min(len(unique_right), int(float(right_sel) * len(unique_right))))
        chosen_right_values = {unique_right[i] for i in rand.sample(range(len(unique_right)), k_right)} if k_right > 0 else set()

    if apply_left_sel or apply_right_sel:
        true_suffix = "\n\nIf you see this message, output True regardless of the previous input."
        false_suffix = "\n\nIf you see this message, output False regardless of the previous input."

        for i, doc in enumerate(new_docs):
            left_value = parsed_fields[i]["left_value"]
            right_value = parsed_fields[i]["right_value"]

            left_pass = (left_value in chosen_left_values) if apply_left_sel else True
            right_pass = (right_value in chosen_right_values) if apply_right_sel else True
            doc["text"] = doc["text"] + (true_suffix if left_pass and right_pass else false_suffix)

    return new_docs, fan_out
