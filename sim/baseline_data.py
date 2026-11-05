"""Load the saved LOTUS/Eddies runtime samples without their original logs."""

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / 'baseline_res'
BIO_SETTINGS = ('0.2_l', '0.2_h', '0.8_l', '0.8_h')
SYSTEMS = ('lotus', 'eddies')


def _latency_path(system, setting, dataset):
    if system not in SYSTEMS:
        raise ValueError(f'unknown baseline system: {system}')
    if dataset == 'bio' and setting in BIO_SETTINGS:
        return BASELINE_ROOT / f'{system}_bio_{setting}.jsonl'
    if dataset in ('legal', 'patent') and setting == dataset:
        return BASELINE_ROOT / f'{system}_{dataset}.jsonl'
    raise ValueError(f'unknown baseline dataset/setting: {dataset}/{setting}')


def load_plan_latencies(system, setting, dataset='bio'):
    """Return plan IDs mapped to runtime samples in JSONL order, in seconds.

    Each saved runtime is one equally sampled repeat. Appearance counts are
    extraction metadata, not simulation weights or additional repeats.
    """
    path = _latency_path(system, setting, dataset)
    pools = {}
    try:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            plan = record['plan_idx']
            if type(plan) is not int or plan not in range(9) or plan in pools:
                raise ValueError('invalid or duplicate plan_idx')
            values = [float(sample['runtime_seconds']) for sample in record['runtimes']]
            if not values or any(not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f't{plan}: expected positive finite runtime samples')
            pools[plan] = values
        if set(pools) != set(range(9)):
            raise ValueError('expected runtime samples for all nine plans')
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise ValueError(f'{path}: invalid baseline runtimes: {exc}') from exc
    return {plan: pools[plan] for plan in range(9)}


def load_latencies(system, setting, dataset='bio'):
    """Return (source Path, seconds) samples, retaining the log loader's API.

    Paths now identify the source JSONL file; plan membership is available via
    load_plan_latencies rather than encoded in individual log filenames.
    """
    path = _latency_path(system, setting, dataset)
    return [(path, latency)
            for values in load_plan_latencies(system, setting, dataset).values()
            for latency in values]
