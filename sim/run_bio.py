"""Run Biodex trace simulations and update a summary CSV.

From the repository root, for example::

    python -m sim.run_bio --loads 10 30 50 70 90 110 150
    python -m sim.run_bio --a 0.8 --b l --loads 30 50 --series seqoflow:srtf pd:srtf

The default trace group is a=0.2, b=h. All available query runs in that group
are used. Loads are offered-load percentages and may exceed 100. Existing runs
are reused unless --overwrite is given.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from sim.load import service_times, simulate

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / 'sim/traces/biodex'
DEFAULT_SERIES = ('seqoflow:fcfs', 'seqoflow:srtf', 'seqoflow:mlfq',
                  'seqoflow:pdb', 'pd:srtf', 'pu:srtf')
SUFFIX = {'seqoflow': '', 'pd': '_pd', 'pu': '_pu'}
NAME = {'seqoflow': 'Seqoflow', 'pd': 'PD', 'pu': 'PU'}
FIELDS = ('a', 'b', 'workload', 'scheduler', 'load_percent', 'seeds',
          'censored_count', 'throughput_qps', 'p50_jct', 'p50_jct_min',
          'p50_jct_max', 'p99_jct', 'p99_jct_min', 'p99_jct_max')


def read_summary(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['workload'] = row['workload'].lower()
        for field in ('seeds', 'censored_count'):
            row[field] = int(float(row[field])) if row.get(field) else None
        for field in ('load_percent', 'throughput_qps', *FIELDS[8:]):
            row[field] = float(row[field]) if row.get(field) else None
    return rows


def write_summary(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)


def load_traces(a, b, workload):
    """Use every available repeat while requiring all nine query plans."""
    paths = []
    for query in range(9):
        matches = sorted(TRACES.glob(f't{query}_run*_{a}_{b}{SUFFIX[workload]}.json'))
        if not matches:
            raise ValueError(f'no traces for t{query}, a={a}, b={b}, {workload} in {TRACES}')
        paths.extend(matches)
    traces = [json.loads(path.read_text()) for path in paths]
    service_times(traces, True)
    return traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loads', type=float, nargs='+', required=True,
                        help='Offered-load percentages, including values above 100')
    parser.add_argument('--a', default='0.2', help='Trace parameter a (default: 0.2)')
    parser.add_argument('--b', default='h', help='Trace parameter b (default: h)')
    parser.add_argument('--series', nargs='+', default=DEFAULT_SERIES,
                        help='Workload:scheduler pairs (default: six comparison series)')
    parser.add_argument('--seeds', type=int, nargs='+', default=(47, 48, 49))
    parser.add_argument('--arrivals', type=int, default=1000,
                        help='Approximate measured arrivals per run (default: 1000)')
    parser.add_argument('--output-dir', type=Path,
                        help='Result directory (default: sim/results/biodex_runs/<a>_<b>)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Rerun existing load/seed combinations')
    args = parser.parse_args()
    if any(not math.isfinite(load) or load <= 0 for load in args.loads) or args.arrivals <= 0:
        parser.error('loads and arrivals must be positive')
    if len(set(args.loads)) != len(args.loads) or len(set(args.seeds)) != len(args.seeds):
        parser.error('loads and seeds must not contain duplicates')
    series = [tuple(value.lower().split(':')) for value in args.series]
    if any(len(pair) != 2 or pair[0] not in SUFFIX or
           pair[1] not in ('fcfs', 'srtf', 'mlfq', 'pdb') for pair in series):
        parser.error('series must use seqoflow/pd/pu and fcfs/srtf/mlfq/pdb')
    if len(set(series)) != len(series):
        parser.error('series must not contain duplicates')

    try:
        trace_sets = {workload: load_traces(args.a, args.b, workload)
                      for workload, _ in series}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    output = args.output_dir or ROOT / 'sim/results/biodex_runs' / f'{args.a}_{args.b}'
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'latency_summary.csv'
    existing = {(r['a'], r['b'], r['workload'], r['scheduler'], r['load_percent']): r
                for r in read_summary(summary_path)}
    for workload, scheduler in series:
        traces = trace_sets[workload]
        mean_service = mean(map(sum, service_times(traces, True)))
        for load in sorted(args.loads):
            key = (args.a, args.b, workload, scheduler, load)
            if key in existing and not args.overwrite:
                print(f'{workload} + {scheduler}: {load:g}% already in summary', flush=True)
                continue
            runs = []
            for seed in args.seeds:
                path = raw / f'{args.a}_{args.b}_{NAME[workload]}_{scheduler}_{load:g}_{seed}.json'
                if path.exists() and not args.overwrite:
                    result = json.loads(path.read_text())
                else:
                    result = simulate(
                        traces, scheduler=scheduler,
                        arrival_rate=load / 100 / mean_service,
                        warmup=500 * mean_service,
                        duration=args.arrivals * mean_service / (load / 100),
                        followup=10000 * mean_service, seed=seed,
                        include_profiling=True, alpha=0.1, srtf_smoothing=0.2,
                    )
                    result.update(workload=workload, a=args.a, b=args.b,
                                  requested_load_percent=load)
                    path.write_text(json.dumps(result) + '\n')
                runs.append(result)
                print(f'{workload} + {scheduler}: {load:g}% seed {seed}, '
                      f'censored={result["censored_count"]}', flush=True)
            row = dict(a=args.a, b=args.b, workload=workload, scheduler=scheduler,
                       load_percent=load, seeds=len(runs),
                       censored_count=sum(r['censored_count'] for r in runs),
                       throughput_qps=mean(r['throughput'] for r in runs))
            for metric in ('p50_jct', 'p99_jct'):
                values = [r[metric] for r in runs]
                complete = not row['censored_count'] and all(v is not None for v in values)
                row[metric] = mean(values) if complete else None
                row[metric + '_min'] = min(values) if complete else None
                row[metric + '_max'] = max(values) if complete else None
            existing[key] = row
    summary = sorted(existing.values(), key=lambda r: (
        r['a'], r['b'], r['workload'], r['scheduler'], r['load_percent']))
    write_summary(summary_path, summary)
    print(f'Saved {len(summary)} rows to {summary_path}')


if __name__ == '__main__':
    main()
