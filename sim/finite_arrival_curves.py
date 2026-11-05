"""Plot BioDEX-0.2-h and Legal finite-arrival latencies in a 2x2 figure.

Rows are BioDEX-0.2-h and Legal; columns are P50 and P99.

Run after ``sim.finite_arrivals`` has written its summaries::

    python -m sim.finite_arrival_curves

Missing panels are shown as empty until their finite-arrival runs exist.
By default, use the task count covering the most panels and runs. Pass --tasks
to select a specific count. Summary rows and raw results are merged by run.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean

# from sim.finite_arrivals import METHODS
METHODS = {'seqoflow:srtf': ('seqoflow', 'srtf'), 'seqoflow:fcfs': ('seqoflow', 'fcfs'), 'seqoflow:mlfq': ('seqoflow', 'mlfq'), 'seqoflow:pdb': ('seqoflow', 'pdb'), 'pd:srtf': ('pd', 'srtf'), 'pu:srtf': ('pu', 'srtf'), 'lotus:fcfs': ('lotus', 'fcfs'), 'eddies:fcfs': ('eddies', 'fcfs')}

ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    ('BioDEX - (0.2, H)', 'bio', '0.2_h'),
    ('Legal', 'legal', 'legal'),
)
STYLES = {
    'seqoflow:srtf': ('SF:SRTF', '#0e9aa7'),
    'seqoflow:fcfs': ('SF:FCFS', '#f39120'),
    'seqoflow:mlfq': ('SF:MLFQ', '#79c39e'),
    'seqoflow:pdb': ('SF:PDB', '#cfb97e'),
    'pd:srtf': ('Static-1:SRTF', '#F12C25'),
    'pu:srtf': ('Static-2:SRTF', "#D3DB67"),
    'lotus:fcfs': ('LOTUS:FCFS', "#F26A3D"),
    'eddies:fcfs': ('Eddies:FCFS', "#FFC64F"),
    #d8cf00, FFC64F
}
REQUIRED = {'dataset', 'setting', 'method', 'arrival_rate_qps', 'seed',
            'tasks', 'p50_jct', 'p99_jct'}


def load_panels(input_dir, tasks=None):
    """Merge summaries/raw runs, then select one task count across all panels."""
    panels = []
    for title, dataset, setting in PANELS:
        summary = input_dir / dataset / setting / 'latency_summary.csv'
        rows = {}

        def add_row(row, source):
            try:
                if (row['dataset'], row['setting']) != (dataset, setting):
                    raise ValueError('unexpected dataset or setting')
                count, seed = int(row['tasks']), int(row['seed'])
                rate = float(row['arrival_rate_qps'])
                p50, p99 = float(row['p50_jct']), float(row['p99_jct'])
                if (count < 1 or row['method'] not in METHODS
                        or not math.isfinite(rate) or rate <= 0
                        or not all(math.isfinite(value) and value >= 0 for value in (p50, p99))):
                    raise ValueError('invalid method, task count, rate, or latency')
                rows[row['method'], rate, seed, count] = {
                    'method': row['method'], 'rate': rate, 'seed': seed, 'tasks': count,
                    'p50_jct': p50, 'p99_jct': p99,
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'{source}: invalid finite-arrival result: {exc}') from exc

        if summary.exists():
            with summary.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if not REQUIRED <= set(reader.fieldnames or ()):
                    raise ValueError(f'{summary}: missing finite-arrival summary columns')
                for row in reader:
                    add_row(row, summary)
        # Raw files recover rows omitted by older summary overwrites and take
        # precedence over a stale CSV entry for the same run and task count.
        for path in sorted((summary.parent / 'raw').glob('*.json')):
            add_row(json.loads(path.read_text()), path)
        panels.append({'title': title, 'rows': list(rows.values())})

    if tasks is None:
        coverage = defaultdict(set)
        run_counts = defaultdict(int)
        for index, panel in enumerate(panels):
            for row in panel['rows']:
                coverage[row['tasks']].add(index)
                run_counts[row['tasks']] += 1
        if coverage:
            tasks = max(coverage, key=lambda count: (
                len(coverage[count]), run_counts[count], count))
    for panel in panels:
        panel['rows'] = [row for row in panel['rows'] if row['tasks'] == tasks]
    return panels


def series_points(rows, metric):
    """Mean per-seed percentile with min/max across seeds at each rate."""
    groups = defaultdict(list)
    for row in rows:
        groups[row['method'], row['rate']].append(row[metric])
    return {
        method: [(rate, mean(values), min(values), max(values))
                 for (name, rate), values in sorted(groups.items(), key=lambda item: item[0][1])
                 if name == method]
        for method in METHODS
    }


def plot_curves(panels, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({
        'font.size': 10, 'axes.labelsize': 12, 'axes.titlesize': 12,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 11, 'figure.dpi': 300,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True)
    handles = {}
    plots = [(panel, metric, percentile) for panel in panels
             for metric, percentile in (('p50_jct', 'P50'), ('p99_jct', 'P99'))]
    k = 1 + len(METHODS) * len(plots)
    for ax, (panel, metric, percentile) in zip(axes.ravel(), plots):
        points_by_method = series_points(panel['rows'], metric)
        for method in METHODS:
            points = points_by_method[method]
            if not points:
                continue
            label, color = STYLES[method]
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            yerr = [[max(0.0, point[1] - point[2]) for point in points],
                    [max(0.0, point[3] - point[1]) for point in points]]
            line = ax.errorbar(
                x, y, yerr=yerr, label=label, color=color, linestyle='-',
                marker='D', markersize=4.8, linewidth=1.5,
                ecolor=to_rgba(color, 0.28), elinewidth=0.7,
                capthick=0.7, capsize=2.5,
                **({'markerfacecolor': 'white', 'mew': 1.6}
                   if metric == 'p99_jct' else {}),
                zorder=k
            )
            if method not in handles:
                handles[method] = line
            k -= 1
        if not panel['rows']:
            ax.text(0.5, 0.5, 'No finite-arrival results',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{panel["title"]} {percentile}', pad=4)
        # ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        import matplotlib.ticker as ticker
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
        ax.xaxis.set_minor_locator(ticker.NullLocator())
        rates = [row['rate'] for row in panel['rows']]
        ax.set_xlim(min(0.001, min(rates) / 1.1) if rates else 0.001,
                    max(1, max(rates) * 1.1) if rates else 1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(visible=True, which='major', axis='both',
                color='gray', alpha=0.16, linewidth=0.6)

    fig.supxlabel('Arrival Rate (queries/s)', y=0.06, fontsize=12)
    fig.supylabel('Latency (s)', x=0.06, fontsize=12)
    shown = [method for method in METHODS if method in handles]
    if shown:
        fig.legend([handles[method] for method in shown],
                   [STYLES[method][0] for method in shown],
                   loc='upper center', ncol=4, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), columnspacing=1.3,
                   handlelength=2.2, handletextpad=0.5)
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.89), pad=0.8, w_pad=1.0, h_pad=0.5)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'latency_vs_arrival_rate.pdf'
    fig.savefig(path, dpi=500, bbox_inches='tight')
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path,
                        default=ROOT / 'sim/results/finite_arrivals')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'sim/results/finite_arrivals/combined')
    parser.add_argument('--tasks', type=int,
                        help='Filter by task count; default: best panel/run coverage')
    args = parser.parse_args()
    if args.tasks is not None and args.tasks < 1:
        parser.error('--tasks must be positive')
    try:
        panels = load_panels(args.input_dir, args.tasks)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not any(panel['rows'] for panel in panels):
        parser.error('no finite-arrival results found for the requested task count')
    selected_tasks = next(row['tasks'] for panel in panels for row in panel['rows'])
    print(f'Using {selected_tasks}-task runs across all panels')
    print(f'Wrote {plot_curves(panels, args.output_dir)}')


if __name__ == '__main__':
    main()
