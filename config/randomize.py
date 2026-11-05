import random
import json
import math
from copy import deepcopy
import logging
import os

logger = logging.getLogger(__name__)

USE_CONFIG = True

def set_use_config(value: bool):
    """Set the process-wide config mode at the build entry point."""
    global USE_CONFIG
    USE_CONFIG = value

def get_use_config() -> bool:
    """Read the shared mode without capturing an import-time boolean."""
    return USE_CONFIG


DEFAULT_BATCH_SIZE = 10
_batch_size = int(os.environ.get('SEM_QUERY_BATCH_SIZE', DEFAULT_BATCH_SIZE))


def set_batch_size(size: int):
    """Set the process-wide batch size at the build entry point."""
    global _batch_size
    if int(size) < 1:
        raise ValueError('batch size must be positive')
    _batch_size = int(size)
    logger.critical("Batch size: %d", _batch_size)


def get_batch_size() -> int:
    return _batch_size


# Product of all selectivity terms is sampled from this range.
TOTAL_SEL_RANGE = [0.02, 0.18]

# 0.0 collapses token counts toward range midpoint, 1.0 uses full range.
DEFAULT_INPUT_TOKEN_VARIANCE = 1.0


# ============== Config randomization settings ==============

# Sigma of Gaussian noise in log-space for selectivity dispersion.
DEFAULT_SEL_VARIANCE = 0.2

# Legacy and regime modes use the optional percentage; crossover uses fixed phases.
FLUCTUATION_MODE = os.environ.get('SEM_QUERY_FLUCTUATION_MODE', 'legacy')
FLUCTUATION_PERCENT = float(os.environ.get('SEM_QUERY_FLUCTUATION_PERCENT', '0'))
# Source batch that controls each crossover's join phase and reprofile boundary.
CROSSOVER_SOURCES = {0: 'bio_2', 1: 'bio_1', 2: 'bio_1',
                     3: 'bio_1', 4: 'bio_1', 5: 'bio_1',
                     6: 'bio_2', 7: 'bio_2', 8: 'bio_1'}

FLUCTUATION_TARGETS = {
    0: ('t0_sem_filter_5', 'bio_2'),
    1: ('t1_sem_filter_10', 'bio_1'),
    2: ('t2_sem_filter_1', 'bio_1'),
    3: ('t3_sem_join_3', 'bio_1'),
    4: ('t4_sem_filter_2', 'bio_1'),
    5: ('t5_sem_filter_4', 'bio_1'),
    6: ('t6_sem_filter_5', 'bio_2'),
    7: ('t7_sem_filter_5', 'bio_2'),
    8: ('t8_sem_join_3', 'bio_2'),
}
# Maximum number of separate B bursts per source in 'burst' mode.
FLUCTUATION_BURSTS = int(os.environ.get('SEM_QUERY_FLUCTUATION_BURSTS', '3'))
FLUCTUATION_INPUT_RANGE = (20, 25)
FLUCTUATION_SELECTIVITY_RANGE = (0.0, 0.1)
REGIME_SEGMENTS = 4
REGIME_SELECTIVE_RANGE = (0.05, 0.15)
REGIME_PERMISSIVE_RANGE = (0.75, 0.85)

# H
LONG_PREFILL = [4096, 5000]
SHORT_PREFILL = [2, 256]

# L
# LONG_PREFILL = [1024, 2048]
# SHORT_PREFILL = [512, 1024]

# M
# LONG_PREFILL = [2048, 4096]
# SHORT_PREFILL = [256, 512]

# ============== Config randomization settings ==============

LONG_DECODE = [1000, 1024]
SHORT_DECODE = [2, 4]

DEFAULT_RANDOM_SEED = 73
rand = random.Random(DEFAULT_RANDOM_SEED)
_config_seed = DEFAULT_RANDOM_SEED


def set_random_seed(seed):
    """Seed configuration randomization for the current workload run."""
    global _config_seed
    rand.seed(seed)
    _config_seed = seed
    logger.critical("Configuration random seed: %d", seed)


def get_config_seed():
    return _config_seed


def crossover_op_configs(configs, dataset, plan_idx):
    """Two deterministic Bio workloads with opposite optimal operator placement."""
    if dataset != 'bio' or plan_idx not in CROSSOVER_SOURCES:
        raise ValueError('Crossover fluctuation supports Bio t0 through t8 only')

    def snapshot(phase):
        result = deepcopy(configs)
        for entry in result:
            args = entry['op_args']
            name = args['op_name']
            kind = entry['op_type']
            if kind == 'sem_filter':
                args.update({'in': 5, 'sel': 1.0})
            elif kind == 'filter':
                args['sel'] = 1.0
            elif kind == 'sem_proj':
                args['in'] = 5

            if plan_idx == 0:
                if name == 't0_sem_filter_5':
                    args.update({'in': 5 if phase == 'A' else 4000,
                                 'sel': 0.0 if phase == 'A' else 1.0})
                elif name == 't0_join_3':
                    args.update({'left_sel': 1.0,
                                 'right_sel': 1.0 if phase == 'A' else 0.0})
            elif plan_idx == 1:
                if name == 't1_sem_filter_10':
                    args.update({'in': 5 if phase == 'A' else 4000,
                                 'sel': 0.0 if phase == 'A' else 1.0})
                elif name == 't1_sem_filter_2':
                    args.update({'in': 4000 if phase == 'A' else 5,
                                 'sel': 1.0 if phase == 'A' else 0.0})
                elif name == 't1_sem_proj_1':
                    args['in'] = 4000 if phase == 'A' else 5
            elif plan_idx == 2:
                if name in {'t2_sem_filter_4', 't2_sem_filter_1'}:
                    first = (name == 't2_sem_filter_4') == (phase == 'A')
                    args.update({'in': 5 if first else 4000,
                                 'sel': 0.0 if first else 1.0})
            elif plan_idx == 3:
                if name == 't3_sem_filter_4':
                    args.update({'in': 32 if phase == 'A' else 10000,
                                 'sel': 0.2 if phase == 'A' else 1.0})
                elif name == 't3_sem_filter_6':
                    args.update({'in': 5, 'sel': max(0.1, 1 / get_batch_size())})
                elif name == 't3_sem_join_3':
                    size = 512 if phase == 'A' else 32
                    args.update({'left_in': size, 'right_in': size,
                                 'left_sel': 1.0 if phase == 'A' else 0.0,
                                 'right_sel': 1.0})
            elif plan_idx == 5 and kind == 'sem_filter':
                if name == 't5_sem_filter_4':
                    args.update({'in': 5 if phase == 'A' else 4000,
                                 'sel': 0.2 if phase == 'A' else 1.0})
                elif name == 't5_sem_filter_1':
                    args.update({'in': 4000 if phase == 'A' else 5,
                                 'sel': 1.0 if phase == 'A' else 0.2})
            elif plan_idx == 4:
                if name == 't4_sem_filter_2':
                    args.update({'in': 5 if phase == 'A' else 10000,
                                 'sel': 0.0 if phase == 'A' else 1.0})
                elif name == 't4_sem_filter_3':
                    args.update({'in': 5, 'sel': max(0.1, 1 / get_batch_size())})
                elif name == 't4_sem_join_1':
                    args.update({'left_in': 32, 'right_in': 32,
                                 'left_sel': 1.0 if phase == 'A' else 0.0,
                                 'right_sel': 0.5 if phase == 'A' else 0.0})
            elif plan_idx == 6:
                if name == 't6_sem_filter_5':
                    args.update({'in': 5 if phase == 'A' else 4000,
                                 'sel': 0.0 if phase == 'A' else 1.0})
                elif name == 't6_sem_filter_4':
                    args.update({'in': 4000 if phase == 'A' else 5,
                                 'sel': 1.0 if phase == 'A' else 0.0})
                elif name == 't6_sem_proj_6':
                    args['in'] = 4000 if phase == 'A' else 5
                elif name == 't6_sem_proj_7':
                    args['in'] = 4000
                elif name == 't6_join_2':
                    args.update({'left_sel': 0.2, 'right_sel': 0.2})
            elif plan_idx == 7:
                if name == 't7_sem_filter_5':
                    args.update({'in': 5 if phase == 'A' else 4000,
                                 'sel': 0.0 if phase == 'A' else 1.0})
                elif name == 't7_sem_filter_3':
                    args.update({'in': 4000 if phase == 'A' else 5,
                                 'sel': 1.0 if phase == 'A' else 0.0})
                elif name == 't7_sem_proj_4':
                    args['in'] = 4000 if phase == 'A' else 5
                elif name == 't7_sem_filter_8':
                    args.update({'in': 4000, 'sel': 1.0})
                elif name == 't7_sem_join_1':
                    args.update({'left_in': 32, 'right_in': 32,
                                 'left_sel': 0.2, 'right_sel': 0.2})
            elif plan_idx == 8:
                if name == 't8_sem_filter_11':
                    args.update({'in': 32, 'sel': 0.1})
                elif name in {'t8_sem_proj_4', 't8_sem_proj_5',
                              't8_sem_proj_6', 't8_sem_proj_7'}:
                    args['in'] = 32
                elif name == 't8_sem_join_3':
                    args.update({'left_in': 32, 'right_in': 32,
                                 'left_sel': 0.5 if phase == 'A' else 0.0,
                                 'right_sel': 1.0 if phase == 'A' else 0.0})
        return result

    return snapshot('A'), snapshot('B')


def fluctuate_op_config(configs):
    """Freeze the configured Bio target's cheap, selective burst settings.

    Use a separate RNG so fluctuation generation cannot change the baseline
    randomization stream. The caller applies this snapshot to source batches.
    """
    changed = deepcopy(configs)
    targets = {name for name, _source in FLUCTUATION_TARGETS.values()}
    for entry in changed:
        args = entry.get('op_args', {})
        name = args.get('op_name')
        if name not in targets:
            continue
        rng = random.Random(f'{get_config_seed()}:{name}:fluctuation')
        input_size = rng.randint(*FLUCTUATION_INPUT_RANGE)
        keep_rate = rng.uniform(*FLUCTUATION_SELECTIVITY_RANGE)
        if entry['op_type'] == 'sem_filter':
            args['in'] = input_size
            args['sel'] = keep_rate
        elif entry['op_type'] == 'sem_join':
            args['left_in'] = args['right_in'] = input_size
            side_rate = math.sqrt(keep_rate)
            args['left_sel'] = args['right_sel'] = side_rate
        else:
            raise ValueError(f'Fluctuation target {name} must be a semantic filter or join')
    return changed


def regime_op_configs(configs, dataset, plan_idx):
    """Build two plan-independent operator configurations for every Bio query.

    A seeded, balanced assignment makes half of each semantic operator type
    cheap in one regime and expensive in the other. Only filters and semantic
    joins receive regime-dependent decisions; projections change cost only.
    """
    kinds = ('sem_filter', 'sem_proj', 'sem_join')
    cohorts = {}
    sizes = [0, 0]
    for kind in kinds:
        names = sorted(entry['op_args']['op_name'] for entry in configs
                       if entry['op_type'] == kind)
        random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{kind}:cohorts').shuffle(names)
        first = 0 if sizes[0] <= sizes[1] else 1
        for index, name in enumerate(names):
            cohort = (first + index) % 2
            cohorts[name] = cohort
            sizes[cohort] += 1

    def snapshot(regime):
        result = deepcopy(configs)
        for entry in result:
            args = entry['op_args']
            name = args['op_name']
            if name not in cohorts:
                continue
            short = cohorts[name] == regime
            role = 'short' if short else 'long'
            rng = random.Random(f'{get_config_seed()}:{dataset}:{plan_idx}:{name}:{role}')
            bounds = SHORT_PREFILL if short else LONG_PREFILL
            size = rng.randint(*bounds)
            if entry['op_type'] in {'sem_filter', 'sem_proj'}:
                args['in'] = size
            else:
                args['left_in'] = args['right_in'] = size
            if entry['op_type'] in {'sem_filter', 'sem_join'}:
                keep_bounds = REGIME_SELECTIVE_RANGE if short else REGIME_PERMISSIVE_RANGE
                keep = rng.uniform(*keep_bounds)
                if entry['op_type'] == 'sem_filter':
                    args['sel'] = keep
                else:
                    args['left_sel'] = args['right_sel'] = math.sqrt(keep)
        return result

    return snapshot(0), snapshot(1), cohorts


def _prod(values):
    p = 1.0
    for v in values:
        p *= v
    return p


def _clamp(value, low, high):
    return max(low, min(high, value))


def _sample_selectivities(sel_count, target_total, sel_variance, min_sel=1e-4, max_sel=1.0):
    if sel_count <= 0:
        return []

    base = target_total ** (1.0 / sel_count)

    if sel_variance <= 0:
        values = [base for _ in range(sel_count)]
    else:
        values = [base * math.exp(rand.gauss(0.0, sel_variance)) for _ in range(sel_count)]

    values = [_clamp(v, min_sel, max_sel) for v in values]

    cur = _prod(values)
    if cur > 0:
        scale = (target_total / cur) ** (1.0 / sel_count)
        values = [_clamp(v * scale, min_sel, max_sel) for v in values]

    for _ in range(3):
        cur = _prod(values)
        if cur <= 0:
            break
        ratio = target_total / cur
        if 0.98 <= ratio <= 1.02:
            break
        flex_idx = [i for i, v in enumerate(values) if min_sel < v < max_sel]
        if not flex_idx:
            break
        scale = ratio ** (1.0 / len(flex_idx))
        for i in flex_idx:
            values[i] = _clamp(values[i] * scale, min_sel, max_sel)

    return values


def _sample_int_with_variance(bounds, variance):
    low, high = bounds
    if low >= high:
        return low

    v = _clamp(variance, 0.0, 1.0)
    mid = (low + high) / 2.0
    raw = rand.uniform(low, high)
    sampled = mid + (raw - mid) * v
    return int(round(_clamp(sampled, low, high)))


def randomize_op_config(
    dataset,
    plan_idx,
    configs=None,
    total_sel_range=TOTAL_SEL_RANGE,
    sel_variance=DEFAULT_SEL_VARIANCE,
    input_token_variance=DEFAULT_INPUT_TOKEN_VARIANCE,
):
    if configs is None:
        from config.config import HOME
        with open(f"{HOME}/config/{dataset}/t{plan_idx}_config.json", "r") as f:
            configs = json.load(f)
    cfgs = deepcopy(configs)

    sel_fields = []
    op_entries = []
    target_total_sel = rand.uniform(*total_sel_range)

    for cfg, conf in zip(cfgs, configs):
        prefill_setting = rand.choice([LONG_PREFILL, SHORT_PREFILL])
        decode_setting = rand.choice([SHORT_DECODE])
        is_semantic = conf['op_type'].startswith('sem_')

        op_entries.append(
            {
                'cfg': cfg,
                'conf': conf,
                'prefill_setting': prefill_setting,
                'decode_setting': decode_setting,
                'is_semantic': is_semantic,
            }
        )

        for sel_key in ('left_sel', 'right_sel', 'sel'):
            if sel_key in conf['op_args']:
                sel_fields.append((cfg, conf['op_args']['op_name'], sel_key))

    semantic_entries = [entry for entry in op_entries if entry['is_semantic']]
    if semantic_entries:
        long_count = len(semantic_entries) // 2
        if len(semantic_entries) % 2 == 1 and rand.random() < 0.5:
            long_count += 1
        short_count = len(semantic_entries) - long_count
        semantic_prefill_settings = [LONG_PREFILL] * long_count + [SHORT_PREFILL] * short_count
        rand.shuffle(semantic_prefill_settings)
        for entry, setting in zip(semantic_entries, semantic_prefill_settings):
            entry['prefill_setting'] = setting

        logger.info(
            f"plan t{plan_idx} | semantic_prefill_split: long={long_count}, short={short_count}"
        )

    for entry in op_entries:
        cfg = entry['cfg']
        conf = entry['conf']
        prefill_setting = entry['prefill_setting']
        decode_setting = entry['decode_setting']

        logger.critical(
            f"Randomizing config for plan t{plan_idx} | op {conf['op_args']['op_name']} | "
            f"prefill: {prefill_setting} | decode: {decode_setting}"
        )

        if 'in' in conf['op_args']:
            cfg['op_args']['in'] = _sample_int_with_variance(prefill_setting, input_token_variance)
        if 'left_in' in conf['op_args']:
            cfg['op_args']['left_in'] = _sample_int_with_variance(prefill_setting, input_token_variance)
        if 'right_in' in conf['op_args']:
            cfg['op_args']['right_in'] = _sample_int_with_variance(prefill_setting, input_token_variance)

        if 'out' in conf['op_args'] and conf['op_type'] not in {'sem_filter', 'sem_join'}:
            cfg['op_args']['out'] = rand.randint(*decode_setting)

    sampled_sels = _sample_selectivities(
        sel_count=len(sel_fields),
        target_total=target_total_sel,
        sel_variance=sel_variance,
        min_sel=1e-4,
        max_sel=1.0,
    )

    for (cfg, op_name, sel_key), sel_value in zip(sel_fields, sampled_sels):
        cfg['op_args'][sel_key] = sel_value
        logger.critical(f"{op_name} | {sel_key}={sel_value:.4f}")

    if sampled_sels:
        logger.critical(
            f"plan t{plan_idx} | target_total_sel={target_total_sel:.4f} | "
            f"actual_total_sel={_prod(sampled_sels):.4f} | sel_variance={sel_variance}"
        )

    return cfgs
