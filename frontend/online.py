"""Fire Bio t0-t8 queries online, at Poisson arrival times, into the scheduler.

As in sim.load.simulate, inter-arrival gaps and plan choices come from separate
streams seeded with seed and seed + 1, and plans are drawn uniformly. Arrivals
fire at absolute times from the start, so building one query does not delay
the next. Each plan uses the operator configuration that a single-query run
with the same seed uses. --load converts an offered-load percentage into a rate
with the mean service time of the sim's Biodex traces, as sim.run_bio does.

From the repository root, for example::

    python -m frontend.online --rate 0.02 --n-queries 50 --seed 47
    python -m frontend.online --load 70 --n-queries 50 --seed 47 --dry-run

The LLM API key is read from SEM_QUERY_LLM_API_KEY.
"""

import argparse
import logging
import os
import random
import threading
import time
from statistics import mean

from config.config import prepare_scan_config, set_active_plan_config
from config.randomize import (DEFAULT_RANDOM_SEED, get_batch_size, get_use_config,
                              randomize_op_config, set_batch_size, set_random_seed,
                              set_use_config)
from core.exec import Query, Scheduler
from config.models import model_args
from core.llm import VLLMWrapper
from core.profiler import init_profiler
from core.runtime_dump import configure_build_dump
from frontend.build import HOME, _load_candidates, build_static

logger = logging.getLogger(__name__)

DATASET = 'bio'
PLANS = tuple(range(9))


def arrival_schedule(rate, n_queries, seed, plans=PLANS):
    """[(arrival time in seconds from start, plan index)], drawn like sim.load."""
    arrivals_rng = random.Random(seed)
    plans_rng = random.Random(seed + 1)
    schedule, now = [], 0.0
    for _ in range(n_queries):
        now += arrivals_rng.expovariate(rate)
        schedule.append((now, plans[plans_rng.randrange(len(plans))]))
    return schedule


def bio_mean_service(a, b):
    """Mean SeqoFlow service time (exec + profiling) of the sim's Biodex traces."""
    from sim.load import service_times
    from sim.run_bio import load_traces
    return mean(map(sum, service_times(load_traces(a, b, 'seqoflow'), True)))


def configure_plans(plans, seed):
    """Give each plan the configuration a single-query run with this seed uses."""
    for plan_idx in plans:
        if get_use_config():
            set_random_seed(seed)
            set_active_plan_config(DATASET, plan_idx, randomize_op_config(DATASET, plan_idx))
        else:
            prepare_scan_config(DATASET, plan_idx)


def run_online(schedule, llm, seed, scheduler):
    """Serve the schedule; the scheduler must be built after init_profiler."""
    plans = sorted({plan_idx for _, plan_idx in schedule})
    generated = build_static(dataset=DATASET, idx=plans)
    configure_plans(plans, seed)
    worker = threading.Thread(target=scheduler.serve, kwargs={'llm': llm}, daemon=True)
    worker.start()
    logger.info(f"Online arrivals started | queries={len(schedule)} | plans={plans}")

    start = time.perf_counter()
    for qid, (arrival, plan_idx) in enumerate(schedule):
        # Build before the arrival time so that building does not delay it.
        query = Query(qid=qid, db_name=DATASET, plan_idx=plan_idx)
        query.init(logical_plan_candidates=_load_candidates(f't{plan_idx}', generated[f't{plan_idx}']),
                   llm=llm)
        time.sleep(max(0.0, start + arrival - time.perf_counter()))
        scheduler.add_query(query)
        lag = time.perf_counter() - start - arrival
        logger.info(f"Added query qid={qid} from plan=t{plan_idx} at t={arrival:.3f}s (lag {lag:.3f}s)")

    logger.info("Finished query arrivals; waiting for scheduler queue to drain")
    while True:
        with scheduler._cv:
            if not scheduler.queue:
                break
        if not worker.is_alive():
            raise RuntimeError('Scheduler worker exited with queries still queued')
        time.sleep(0.2)
    scheduler.stop()
    worker.join(timeout=5)
    logger.info(f"All {len(schedule)} queries completed in {time.perf_counter() - start:.3f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    arrivals = parser.add_mutually_exclusive_group(required=True)
    arrivals.add_argument('--rate', type=float, help='Poisson arrival rate in queries/second')
    arrivals.add_argument('--load', type=float,
                          help='offered-load percentage, converted with the trace mean service time')
    parser.add_argument('--a', default='0.2', help='trace group a for --load (default: 0.2)')
    parser.add_argument('--b', default='h', help='trace group b for --load (default: h)')
    parser.add_argument('--n-queries', type=int, required=True)
    parser.add_argument('--plans', nargs='+', type=int, default=list(PLANS),
                        help='Bio plan indices to draw from (default: 0-8)')
    parser.add_argument('--seed', type=int, default=DEFAULT_RANDOM_SEED,
                        help='seed for arrivals, plan choice and operator configuration')
    parser.add_argument('--do_cfg', type=int, choices=[0, 1], default=1,
                        help='1 randomizes operator config; 0 uses only static scan row ranges')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='rows per source batch (default: SEM_QUERY_BATCH_SIZE or 10)')
    parser.add_argument('--fully-pipelined', action='store_true')
    parser.add_argument('--pipeline-profile-interval', type=int, default=1)
    parser.add_argument('--scheduler', choices=('srtf', 'fcfs'), default='srtf')
    parser.add_argument('--srtf-alpha', type=float, default=0.1)
    parser.add_argument('--srtf-smoothing', type=float, default=0.2)
    # Defaults come from the 'llm' and 'slm' profiles in models.json.
    parser.add_argument('--llm-model', default=None)
    parser.add_argument('--llm-url', default=None)
    parser.add_argument('--slm-model', default=None)
    parser.add_argument('--slm-url', default=None)
    parser.add_argument('--dry-run', action='store_true',
                        help='print the arrival schedule without building or running queries')
    args = parser.parse_args()
    if args.n_queries < 1:
        parser.error('--n-queries must be positive')
    if not args.plans or any(p not in PLANS for p in args.plans):
        parser.error('--plans must be Bio plan indices 0-8')
    if args.pipeline_profile_interval < 1:
        parser.error('--pipeline-profile-interval must be at least 1')
    if args.pipeline_profile_interval != 1 and not args.fully_pipelined:
        parser.error('--pipeline-profile-interval requires --fully-pipelined')

    if args.rate is not None:
        rate = args.rate
    else:
        service = bio_mean_service(args.a, args.b)
        rate = args.load / 100 / service
        logger.info(f"Load {args.load:g}% of mean service {service:.3f}s -> {rate:.5f} q/s")
    if not rate > 0:
        parser.error('the arrival rate must be positive')
    schedule = arrival_schedule(rate, args.n_queries, args.seed, args.plans)
    logger.info(f"Arrival rate {rate:.5f} q/s, seed {args.seed}")
    if args.dry_run:
        for qid, (arrival, plan_idx) in enumerate(schedule):
            print(f'qid={qid} t={arrival:.3f}s plan=t{plan_idx}')
        return

    set_use_config(args.do_cfg == 1)
    set_random_seed(args.seed)  # also seeds runtime burst schedules
    set_batch_size(get_batch_size() if args.batch_size is None else args.batch_size)
    llm_args, slm_args = model_args('llm'), model_args('slm')
    llm_args.update({k: v for k, v in (('model_name', args.llm_model), ('url', args.llm_url),
                                       ('api_key', os.environ.get('SEM_QUERY_LLM_API_KEY')))
                     if v is not None})
    slm_args.update({k: v for k, v in (('model_name', args.slm_model), ('url', args.slm_url))
                     if v is not None})
    llm, slm = VLLMWrapper(**llm_args), VLLMWrapper(**slm_args)
    configure_build_dump(DATASET, [f'{HOME}/data/{DATASET}/t{p}.json' for p in sorted(set(args.plans))],
                         args.seed, rate, args.n_queries, llm, slm,
                         fully_pipelined=args.fully_pipelined,
                         pipeline_profile_interval=args.pipeline_profile_interval)
    init_profiler(model=slm)
    scheduler = Scheduler(fully_pipelined=args.fully_pipelined,
                          pipeline_profile_interval=args.pipeline_profile_interval,
                          srtf_alpha=args.srtf_alpha, srtf_smoothing=args.srtf_smoothing,
                          policy=args.scheduler)
    run_online(schedule, llm, args.seed, scheduler)


if __name__ == '__main__':
    main()
