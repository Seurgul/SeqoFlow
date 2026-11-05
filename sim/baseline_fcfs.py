"""Run FCFS load simulations from LOTUS and Eddies completed-run latencies.

Each saved runtime in baseline_res supplies one run-to-completion service time
in seconds. There are no scheduling steps or estimates.

Example, from the repository root::

    python -m sim.baseline_fcfs --loads 30 50 70 90
    python -m sim.baseline_fcfs --dataset legal --loads 30 50 70 90
"""

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from sim.baseline_data import BIO_SETTINGS, ROOT, SYSTEMS, load_latencies
from sim.load import simulate


FIELDS = ('system', 'setting', 'scheduler', 'load_percent', 'seeds',
          'service_samples', 'mean_service_seconds', 'censored_count',
          'throughput_qps', 'p50_jct', 'p50_jct_min', 'p50_jct_max',
          'p99_jct', 'p99_jct_min', 'p99_jct_max')


def read_summary(path):
    if not path.exists():
        return {}
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {(row['system'], row['setting'], float(row['load_percent'])): row
            for row in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('bio', 'legal', 'patent'), default='bio')
    parser.add_argument('--loads', type=float, nargs='+', required=True,
                        help='Offered load percentages; values above 100 are allowed')
    parser.add_argument('--settings', nargs='+', choices=BIO_SETTINGS,
                        help='BioDEX settings (default: all four)')
    parser.add_argument('--systems', nargs='+', choices=SYSTEMS,
                        default=SYSTEMS)
    parser.add_argument('--seeds', type=int, nargs='+', default=(47, 48, 49))
    parser.add_argument('--arrivals', type=int, default=1000,
                        help='Approximate measured arrivals per run')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.dataset != 'bio' and args.settings is not None:
        parser.error('--settings applies only to --dataset bio')
    settings = (args.settings or BIO_SETTINGS) if args.dataset == 'bio' else (args.dataset,)
    output_dir = args.output_dir or ROOT / f'sim/results/baselines_{args.dataset}'
    if (args.arrivals <= 0 or any(not math.isfinite(x) or x <= 0 for x in args.loads)
            or len(set(args.loads)) != len(args.loads)
            or len(set(args.seeds)) != len(args.seeds)
            or len(set(settings)) != len(settings)
            or len(set(args.systems)) != len(args.systems)):
        parser.error('arrivals and loads must be positive; values must not repeat')

    # Validate all requested inputs before creating any output.
    try:
        inputs = {(system, setting): load_latencies(system, setting, args.dataset)
                  for system in args.systems for setting in settings}
    except ValueError as exc:
        parser.error(str(exc))

    raw = output_dir / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'latency_summary.csv'
    rows = read_summary(summary_path)
    for (system, setting), samples in inputs.items():
        average = mean(latency for _, latency in samples)
        # A completed baseline run is represented as one nonpreemptible FCFS job.
        jobs = [[{'exec': latency}] for _, latency in samples]
        for load in sorted(args.loads):
            key = (system, setting, load)
            if key in rows and not args.overwrite:
                print(f'{system} {setting} {load:g}% already in summary', flush=True)
                continue
            runs = []
            for seed in args.seeds:
                path = raw / f'{system}_{setting}_fcfs_{load:g}_{seed}.json'
                if path.exists() and not args.overwrite:
                    result = json.loads(path.read_text())
                else:
                    result = simulate(
                        jobs, scheduler='fcfs',
                        arrival_rate=load / 100 / average,
                        warmup=500 * average,
                        duration=args.arrivals * average / (load / 100),
                        followup=10000 * average, seed=seed,
                    )
                    result.update(system=system, setting=setting,
                                  requested_load_percent=load,
                                  service_samples=len(samples),
                                  source_logs=[str(p.relative_to(ROOT)) for p, _ in samples])
                    path.write_text(json.dumps(result) + '\n')
                runs.append(result)
                print(f'{system} {setting} FCFS {load:g}% seed {seed}: '
                      f'arrivals={result["measurement_arrivals"]} '
                      f'censored={result["censored_count"]}', flush=True)
            row = dict(system=system, setting=setting, scheduler='fcfs',
                       load_percent=load, seeds=len(runs),
                       service_samples=len(samples), mean_service_seconds=average,
                       censored_count=sum(r['censored_count'] for r in runs),
                       throughput_qps=mean(r['throughput'] for r in runs))
            for metric in ('p50_jct', 'p99_jct'):
                values = [r[metric] for r in runs]
                complete = not row['censored_count'] and all(v is not None for v in values)
                row[metric] = mean(values) if complete else None
                row[metric + '_min'] = min(values) if complete else None
                row[metric + '_max'] = max(values) if complete else None
            rows[key] = row
    with summary_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS}
                         for _, row in sorted(rows.items()))
    print(f'Saved {len(rows)} rows to {summary_path}')


if __name__ == '__main__':
    main()
