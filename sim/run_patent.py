"""Run legal trace simulations and update the shared summary CSV.

Example (with Matplotlib available):
    python -m sim.run_legal --loads 10 30 50 70 90 110 150
    python -m sim.run_legal --loads 110 150 --series seqoflow:srtf pd:srtf --output-dir sim/results/high_load

Loads are offered-load percentages and may exceed 100. Results are saved under
--output-dir; existing runs are reused unless --overwrite is given.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean

from sim.load import service_times, simulate

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / 'sim/traces/patent'
DEFAULT_SERIES = ('seqoflow:fcfs', 'seqoflow:srtf', 'seqoflow:mlfq',
                  'seqoflow:pdb', 'pd:srtf', 'pu:srtf')
SUFFIX = {'seqoflow': '', 'pd': '_pd', 'pu': '_pu'}
NAME = {'seqoflow': 'Seqoflow', 'pd': 'PD', 'pu': 'PU'}
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#6B4C9A', '#7A5C38')
FIELDS = ('workload', 'scheduler', 'load_percent', 'seeds', 'censored_count',
          'throughput_qps', 'p50_jct', 'p50_jct_min', 'p50_jct_max',
          'p99_jct', 'p99_jct_min', 'p99_jct_max')


def read_summary(path):
    if not path.exists():
        return []
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['workload'] = row['workload'].lower()
        for field in ('seeds', 'censored_count'):
            row[field] = int(float(row[field])) if row.get(field) else None
        for field in ('load_percent', 'throughput_qps', *FIELDS[6:]):
            row[field] = float(row[field]) if row.get(field) else None
    return rows


def write_summary(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in FIELDS} for row in rows)


def plot(rows, series, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for axis in ('load', 'throughput'):
        fig, panels = plt.subplots(1, 2, figsize=(15, 6))
        for panel, metric in zip(panels, ('p50_jct', 'p99_jct')):
            for color, (workload, scheduler) in zip(COLORS, series):
                points = [r for r in rows if (r['workload'], r['scheduler']) ==
                          (workload, scheduler)]
                x = [r['load_percent'] if axis == 'load' else r['throughput_qps']
                     for r in points]
                y = [r[metric] for r in points]
                ylim = 0.1 * max(y)
                label = f'{NAME[workload]} + {scheduler.upper()}'
                line, = panel.plot(x, y, marker='o', linewidth=1.7, label=label,
                                   color=color, linestyle='--' if workload != 'seqoflow' else '-')
                panel.fill_between(x, [r[metric + '_min'] if r[metric + '_min'] is not None else math.nan for r in points],
                                   [r[metric + '_max'] if r[metric + '_max'] is not None else math.nan for r in points],
                                   color=line.get_color(), alpha=0.08)
                panel.set_ylim(bottom=0, top=ylim)
            panel.set_title(metric[:3].upper() + ' query latency')
            panel.set_ylabel('Latency (seconds)')
            panel.set_xlabel('Offered load (%)' if axis == 'load'
                             else 'Measured throughput (queries/s)')
            # panel.set_ylim(bottom=0)
            panel.grid(alpha=0.2)
            if axis == 'load':
                panel.set_xticks(sorted({r['load_percent'] for r in rows}))
        fig.suptitle('Legal workloads: latency vs. ' +
                     ('offered load' if axis == 'load' else 'measured throughput'))
        handles, labels = panels[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, 0.92))
        fig.tight_layout(rect=(0, 0, 1, 0.82))
        fig.savefig(output / f'latency_vs_{axis}.png', dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loads', type=float, nargs='+', required=True,
                        help='Offered-load percentages, including values above 100')
    parser.add_argument('--series', nargs='+', default=DEFAULT_SERIES,
                        help='Workload:scheduler pairs (default: six comparison series)')
    parser.add_argument('--seeds', type=int, nargs='+', default=(47, 48, 49))
    parser.add_argument('--arrivals', type=int, default=1000,
                        help='Approximate measured arrivals per run (default: 1000)')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'sim/results/patent_runs')
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

    output = args.output_dir
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    summary_path = output / 'latency_summary.csv'
    existing = {(r['workload'], r['scheduler'], r['load_percent']): r
                for r in read_summary(summary_path)}
    for workload, scheduler in series:
        traces = [json.loads((TRACES / f't{i}{SUFFIX[workload]}.json').read_text())
                  for i in range(9)]
        mean_service = mean(map(sum, service_times(traces, True)))
        for load in sorted(args.loads):
            key = (workload, scheduler, load)
            if key in existing and not args.overwrite:
                print(f'{workload} + {scheduler}: {load:g}% already in summary', flush=True)
                continue
            runs = []
            for seed in args.seeds:
                path = raw / f'{NAME[workload]}_{scheduler}_{load:g}_{seed}.json'
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
                    result.update(workload=workload, requested_load_percent=load)
                    path.write_text(json.dumps(result) + '\n')
                runs.append(result)
                print(f'{workload} + {scheduler}: {load:g}% seed {seed}, '
                      f'censored={result["censored_count"]}', flush=True)
            row = dict(workload=workload, scheduler=scheduler, load_percent=load,
                       seeds=len(runs), censored_count=sum(r['censored_count'] for r in runs),
                       throughput_qps=mean(r['throughput'] for r in runs))
            for metric in ('p50_jct', 'p99_jct'):
                values = [r[metric] for r in runs]
                complete = not row['censored_count'] and all(v is not None for v in values)
                row[metric] = mean(values) if complete else None
                row[metric + '_min'] = min(values) if complete else None
                row[metric + '_max'] = max(values) if complete else None
            existing[key] = row
    summary = sorted(existing.values(), key=lambda r: (
        r['workload'], r['scheduler'], r['load_percent']))
    write_summary(summary_path, summary)
    print(f'Saved {len(summary)} rows to {summary_path}')


if __name__ == '__main__':
    main()
