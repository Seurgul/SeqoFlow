import glob
from pathlib import Path
import logging
import sys
import json
import argparse
import random
import threading
import time
import os

from opt.logic_op import Plan
from opt.memo import get_memoizer
from opt.serialize import load_plan, dump_plan
from config.config import HOME, prepare_scan_config, set_active_plan_config
from config.randomize import (DEFAULT_RANDOM_SEED, get_use_config,
                              randomize_op_config, set_batch_size, set_random_seed,
                              set_use_config, get_batch_size)
from core.exec import Query, Scheduler
from config.models import model_args
from core.llm import VLLMWrapper
from core.profiler import Profiler, get_profiler, init_profiler
from core.runtime_dump import configure_experiment_dump
from frontend.build import _load_static, build_static, build_plan_dict, get_plan_idx

def setup_logging():
    logging.basicConfig(
        level=logging.CRITICAL,
        format="%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
    # logging.getLogger("opt.logic_op").setLevel(logging.WARNING)
    logging.getLogger("opt.memo").setLevel(logging.WARNING)
    logging.getLogger("vllm").setLevel(logging.WARNING)


setup_logging()
logger = logging.getLogger(__name__)


def _configure_plan(dataset, plan_idx, configured_plans):
    """Use one configuration per plan for the lifetime of an experiment run."""
    if plan_idx in configured_plans:
        return
    if get_use_config():
        set_active_plan_config(dataset, plan_idx,
                               randomize_op_config(dataset, plan_idx))
    else:
        prepare_scan_config(dataset, plan_idx)
    configured_plans.add(plan_idx)

def build_run_static(
    dataset=None,
    idx=None,
    llm: VLLMWrapper = None,
    slm: VLLMWrapper = None,
    lam: float = 0.2,
    n_queries: int = 1,
    seed: int = 43,
    full_batch_execution: bool = False,
):
    if idx:
        plans = [f'{HOME}/data/{dataset}/t{i}.json' for i in idx]
    else:
        plans = glob.glob(f"{HOME}/data/{dataset}/*.json")

    rand = random.Random(seed)
    all_plan = [Path(p).stem for p in plans]
    configured_plans = set()

    # profiler = init_profiler(model=slm)
    configure_experiment_dump(dataset, plans, n_queries, 'static',
                              full_batch_execution=full_batch_execution, seed=seed)
    scheduler = Scheduler(profile_enabled=False,
                          full_batch_execution=full_batch_execution)

    worker = threading.Thread(target=scheduler.serve, kwargs={"llm": llm, "prf": False}, daemon=True)
    worker.start()

    logger.info(
        f"Serving scheduler started | lambda={lam:.4f} q/s | total_queries={n_queries}"
    )

    next_qid = 0
    for _ in range(n_queries):
        dt = rand.expovariate(lam)
        time.sleep(dt)

        plan_name = rand.choice(all_plan)
        plan_idx = get_plan_idx(plan_name)
        build_static(dataset=dataset, idx=[plan_idx])
        cand = _load_static(plan_name)
        _configure_plan(dataset, plan_idx, configured_plans)

        query = Query(qid=next_qid, db_name=dataset, plan_idx=plan_idx)
        query.init(logical_plan_candidates=cand, llm=llm)
        scheduler.add_query(query)
        logger.info(
            f"Added query qid={next_qid} from plan={plan_name} after inter-arrival={dt:.3f}s"
        )
        next_qid += 1

    logger.info("Finished query arrivals; waiting for scheduler queue to drain")
    while True:
        with scheduler._cv:
            if not scheduler.queue:
                break
        time.sleep(0.2)

    scheduler.stop()
    worker.join(timeout=5)
    logger.info("Scheduler stopped")

def build_run_pd(
    dataset=None,
    idx=None,
    llm: VLLMWrapper = None,
    slm: VLLMWrapper = None,
    lam: float = 0.2,
    n_queries: int = 1,
    seed: int = 43,
    full_batch_execution: bool = False,
):
    if idx:
        plans = [f'{HOME}/data/{dataset}/handwritten/t{i}_1.json' for i in idx]
    else:
        plans = glob.glob(f"{HOME}/data/{dataset}/handwritten/*.json")

    rand = random.Random(seed)
    all_plan = [Path(p).stem for p in plans]
    configured_plans = set()

    # profiler = init_profiler(model=slm)
    configure_experiment_dump(dataset, plans, n_queries, 'pd',
                              full_batch_execution=full_batch_execution, seed=seed)
    scheduler = Scheduler(profile_enabled=False,
                          full_batch_execution=full_batch_execution)

    worker = threading.Thread(target=scheduler.serve, kwargs={"llm": llm, "prf": False}, daemon=True)
    worker.start()

    logger.info(
        f"Serving scheduler started | lambda={lam:.4f} q/s | total_queries={n_queries}"
    )

    next_qid = 0
    for _ in range(n_queries):
        dt = rand.expovariate(lam)
        time.sleep(dt)

        plan_name = rand.choice(all_plan)
        with open(f"{HOME}/data/{dataset}/handwritten/{plan_name}.json", "r") as f:
            plan_json = json.load(f)
        plan_dict = build_plan_dict(plan_json)
        os.makedirs(f"{HOME}/plans", exist_ok=True)
        with open(f"{HOME}/plans/{plan_name}.json", "w") as f:
            json.dump({
                "schema_version": 1,
                "plan": plan_dict
            }, f, indent=4)
        plan = Plan(load_plan(f"{HOME}/plans/{plan_name}.json"))
        plan.populate_props()
        cand = [plan]

        plan_idx = get_plan_idx(plan_name)
        _configure_plan(dataset, plan_idx, configured_plans)

        query = Query(qid=next_qid, db_name=dataset, plan_idx=plan_idx)
        query.init(logical_plan_candidates=cand, llm=llm)
        scheduler.add_query(query)
        logger.info(
            f"Added query qid={next_qid} from plan={plan_name} after inter-arrival={dt:.3f}s"
        )
        next_qid += 1

    logger.info("Finished query arrivals; waiting for scheduler queue to drain")
    while True:
        with scheduler._cv:
            if not scheduler.queue:
                break
        time.sleep(0.2)

    scheduler.stop()
    worker.join(timeout=5)
    logger.info("Scheduler stopped")

def build_run_pu(
    dataset=None,
    idx=None,
    llm: VLLMWrapper = None,
    slm: VLLMWrapper = None,
    lam: float = 0.2,
    n_queries: int = 1,
    seed: int = 43,
    full_batch_execution: bool = False,
):
    if idx:
        plans = [f'{HOME}/data/{dataset}/handwritten/t{i}_2.json' for i in idx]
    else:
        plans = glob.glob(f"{HOME}/data/{dataset}/handwritten/*.json")

    rand = random.Random(seed)
    all_plan = [Path(p).stem for p in plans]
    configured_plans = set()

    # profiler = init_profiler(model=slm)
    configure_experiment_dump(dataset, plans, n_queries, 'pu',
                              full_batch_execution=full_batch_execution, seed=seed)
    scheduler = Scheduler(profile_enabled=False,
                          full_batch_execution=full_batch_execution)

    worker = threading.Thread(target=scheduler.serve, kwargs={"llm": llm, "prf": False}, daemon=True)
    worker.start()

    logger.info(
        f"Serving scheduler started | lambda={lam:.4f} q/s | total_queries={n_queries}"
    )

    next_qid = 0
    for _ in range(n_queries):
        dt = rand.expovariate(lam)
        time.sleep(dt)

        plan_name = rand.choice(all_plan)
        with open(f"{HOME}/data/{dataset}/handwritten/{plan_name}.json", "r") as f:
            plan_json = json.load(f)
        plan_dict = build_plan_dict(plan_json)
        os.makedirs(f"{HOME}/plans", exist_ok=True)
        with open(f"{HOME}/plans/{plan_name}.json", "w") as f:
            json.dump({
                "schema_version": 1,
                "plan": plan_dict
            }, f, indent=4)
        plan = Plan(load_plan(f"{HOME}/plans/{plan_name}.json"))
        plan.populate_props()
        cand = [plan]

        plan_idx = get_plan_idx(plan_name)
        _configure_plan(dataset, plan_idx, configured_plans)

        query = Query(qid=next_qid, db_name=dataset, plan_idx=plan_idx)
        query.init(logical_plan_candidates=cand, llm=llm)
        scheduler.add_query(query)
        logger.info(
            f"Added query qid={next_qid} from plan={plan_name} after inter-arrival={dt:.3f}s"
        )
        next_qid += 1

    logger.info("Finished query arrivals; waiting for scheduler queue to drain")
    while True:
        with scheduler._cv:
            if not scheduler.queue:
                break
        time.sleep(0.2)

    scheduler.stop()
    worker.join(timeout=5)
    logger.info("Scheduler stopped")

def execution_llm():
    """The execution model shared by the SeqoFlow baselines and LOTUS eddies."""
    return VLLMWrapper(**model_args("llm"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['bio', 'arxiv', 'legal', 'patent'])
    parser.add_argument("--do_cfg", type=int, choices=[0, 1], default=1,
                        help="1 randomizes operator config; 0 uses only static scan row ranges")
    parser.add_argument("--plan", nargs='+', type=int, required=False)
    parser.add_argument("--mode", type=str, choices=["static", "pd", "pu"], default="static")
    parser.add_argument("--full-batch-execution", action="store_true",
                        help="evaluate filters while passing full batches downstream")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="seed for workload and operator-configuration randomization",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                        help="rows per source batch (default: SEM_QUERY_BATCH_SIZE or 10)")

    args = parser.parse_args()
    dataset = args.dataset
    plan_idx = args.plan
    set_use_config(args.do_cfg == 1)
    set_random_seed(args.seed)
    set_batch_size(get_batch_size() if args.batch_size is None else args.batch_size)

    # Profiles live in models.json (gitignored; template: models.example.json).
    # slm = VLLMWrapper(**model_args("slm-qwen3-4b-8001"))
    # llm = VLLMWrapper(**model_args("llm-qwen2.5-32b"))
    # llm = VLLMWrapper(**model_args("llm-qwen3-8b"))
    llm = execution_llm()
    # llm = VLLMWrapper(**model_args("llm-deepseek-flash"))


    if args.mode == "static":
        build_run_static(
            dataset=dataset,
            idx=plan_idx,
            lam=0.5,
            llm=llm,
            seed=args.seed,
            full_batch_execution=args.full_batch_execution,
            # slm=slm,
        )
    elif args.mode == "pd":
        build_run_pd(
            dataset=dataset,
            idx=plan_idx,
            lam=0.5,
            llm=llm,
            seed=args.seed,
            full_batch_execution=args.full_batch_execution,
            # slm=slm,
        )
    elif args.mode == "pu":
        build_run_pu(
            dataset=dataset,
            idx=plan_idx,
            lam=0.5,
            llm=llm,
            seed=args.seed,
            full_batch_execution=args.full_batch_execution,
            # slm=slm,
        )
