import glob
from pathlib import Path
import logging
import sys
import json
import argparse
import random
import threading
import time

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
from core.runtime_dump import configure_build_dump


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
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

def get_plan_idx(plan_name: str):
    import re
    match = re.search(r't(\d+)_', plan_name)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r't(\d+)$', plan_name)
        if match:
            return int(match.group(1))
        else:
            raise ValueError(f"Plan name {plan_name} does not contain a valid plan index.")

def build_plan_dict(raw_plan: dict) -> dict:
    nodes = raw_plan["nodes"]

    referenced_nodes = set()
    for node in nodes.values():
        for ref in node.get("refs", {}).values():
            referenced_nodes.add(str(ref))

    root_ids = [node_id for node_id in nodes.keys() if node_id not in referenced_nodes]
    if len(root_ids) != 1:
        raise ValueError(f"Expected exactly one root node, found {root_ids}")

    def build_node(node_id: str) -> dict:
        node = nodes[node_id]
        plan_node = {
            "oid": node["oid"],
            "op_type": node["op_type"],
            "requires": node.get("requires", []),
            "yields": node.get("yields", []),
            "params": node.get("params", {}),
        }

        refs = node.get("refs", {})
        if "left" in refs:
            plan_node["left"] = build_node(str(refs["left"]))
        if "right" in refs:
            plan_node["right"] = build_node(str(refs["right"]))
        if "child" in refs:
            plan_node["child"] = build_node(str(refs["child"]))

        return plan_node

    return build_node(root_ids[0])


def create_workspace(plan_file: str):
    plan_path = Path(plan_file)
    workspace_dir = HOME / "plans" / plan_path.stem
    destination = workspace_dir / plan_path.name
    plan_name = plan_path.stem

    workspace_dir.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "r") as src, open(destination, "w") as dst:
        raw_plan = json.load(src)
        plan_dict = build_plan_dict(raw_plan)
        json.dump({
            "schema_version": 1,
            "plan": plan_dict
        }, dst, indent=4)

    return plan_name, destination


def build_static(dataset, idx=None):
    if idx:
        plans = [f'{HOME}/data/{dataset}/t{i}.json' for i in idx]
    else:
        plans = glob.glob(f"{HOME}/data/{dataset}/*.json")
    generated = {}
    for plan_file in plans:
        plan_name, path = create_workspace(plan_file)
        plan = Plan(load_plan(path))
        # plan.assign_oid()
        plan.populate_props()
        plan.pushup(plan.root)
        dump_plan(plan.root, f"{HOME}/plans/{plan_name}/{plan_name}_pu.json")
        plan.pushdown()
        memo = get_memoizer()
        logger.critical(f"{plan_name} | {len(memo._memo)} plans captured")
        generated[plan_name] = []
        for i, plan in enumerate(memo._memo):
            plan.populate_props()
            path = f"{HOME}/plans/{plan_name}/{plan_name}_{i}.json"
            dump_plan(plan.root, path)
            generated[plan_name].append(path)
        memo.clear()
    return generated


def _load_candidates(plan_name: str, cand_files=None):
    if cand_files is None:
        cand_files = glob.glob(f"{HOME}/plans/{plan_name}/{plan_name}_*.json")
    cand = [Plan(load_plan(f)) for f in cand_files if not f.endswith("_pu.json")]
    if not cand:
        raise RuntimeError(f"No plan candidates found for {plan_name}")
    return cand

def _load_static(plan_name: str):
    cand_files = glob.glob(f"{HOME}/plans/{plan_name}/{plan_name}_*.json")
    cand = [Plan(load_plan(f)) for f in cand_files if not f.endswith("_pu.json")]
    rand = random.Random(123)
    idx = rand.randint(0, len(cand) - 1)
    if plan_name == "t9":
        idx = 5
    elif plan_name == "t8":
        idx = 3
    cand = [cand[idx]]  # randomly select one candidate for static build, not the best one
    return cand


def build_run(
    dataset=None,
    idx=None,
    llm: VLLMWrapper = None,
    slm: VLLMWrapper = None,
    lam: float = 0.2,
    n_queries: int = 10,
    seed: int = 43,
    full_batch_execution: bool = False,
    fully_pipelined: bool = False,
    pipeline_profile_interval: int = 1,
    profile_budget: int = 20,
):
    if idx:
        plans = [f'{HOME}/data/{dataset}/t{i}.json' for i in idx]
    else:
        plans = glob.glob(f"{HOME}/data/{dataset}/*.json")

    rand = random.Random(seed)
    all_plan = [Path(p).stem for p in plans]
    configured_plans = set()

    configure_build_dump(dataset, plans, seed, lam, n_queries, llm, slm, mode=None,
                         full_batch_execution=full_batch_execution,
                         fully_pipelined=fully_pipelined,
                         pipeline_profile_interval=pipeline_profile_interval)
    profiler = init_profiler(model=slm, profile_budget=profile_budget)
    scheduler = Scheduler(full_batch_execution=full_batch_execution,
                          fully_pipelined=fully_pipelined,
                          pipeline_profile_interval=pipeline_profile_interval)

    worker = threading.Thread(target=scheduler.serve, kwargs={"llm": llm}, daemon=True)
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
        generated = build_static(dataset=dataset, idx=[plan_idx])
        cand = _load_candidates(plan_name, generated[plan_name])
        if plan_idx not in configured_plans:
            if get_use_config():
                set_active_plan_config(dataset, plan_idx,
                                       randomize_op_config(dataset, plan_idx))
            else:
                prepare_scan_config(dataset, plan_idx)
            configured_plans.add(plan_idx)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['bio', 'arxiv', 'legal', 'patent'])
    parser.add_argument("--plan", nargs='+', type=int, required=True)
    parser.add_argument("--do_cfg", type=int, choices=[0, 1], default=1,
                        help="1 randomizes operator config; 0 uses only static scan row ranges")
    parser.add_argument("--full-batch-execution", action="store_true",
                        help="evaluate filters but pass all rows downstream during execution")
    parser.add_argument("--fully-pipelined", action="store_true",
                        help="profile scheduled next steps concurrently with current execution")
    parser.add_argument("--pipeline-profile-interval", type=int, default=1,
                        help="profile every Nth query step in pipelined mode (default: 1)")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="seed for workload and operator-configuration randomization",
    )
    parser.add_argument("--batch-size", type=int, default=None,
                        help="rows per source batch (default: SEM_QUERY_BATCH_SIZE or 10)")
    parser.add_argument("--profile-budget", type=int, default=20,
                        help="model evaluations per profiling probe, shared by the semantic "
                             "operators; at least their count (default: 20)")
    args = parser.parse_args()
    if args.profile_budget < 1:
        parser.error('--profile-budget must be positive')
    if args.pipeline_profile_interval < 1:
        parser.error('--pipeline-profile-interval must be at least 1')
    if args.pipeline_profile_interval != 1 and not args.fully_pipelined:
        parser.error('--pipeline-profile-interval requires --fully-pipelined')
    dataset = args.dataset
    plan_idx = args.plan
    set_use_config(args.do_cfg == 1)
    set_random_seed(args.seed)
    set_batch_size(get_batch_size() if args.batch_size is None else args.batch_size)

    # Profiles live in models.json (gitignored; template: models.example.json).
    slm = VLLMWrapper(**model_args("slm"))
    # slm = VLLMWrapper(**model_args("slm-qwen2.5-1.5b"))
    # llm = VLLMWrapper(**model_args("llm-qwen2.5-32b"))
    llm = VLLMWrapper(**model_args("llm"))
    # llm = VLLMWrapper(**model_args("llm-deepseek-flash"))
    # llm = VLLMWrapper(**model_args("llm-qwen3-8b"))

    build_run(
        dataset=dataset,
        idx=plan_idx,
        lam=0.5,
        n_queries=1,
        llm=llm,
        slm=slm,
        seed=args.seed,
        full_batch_execution=args.full_batch_execution,
        fully_pipelined=args.fully_pipelined,
        pipeline_profile_interval=args.pipeline_profile_interval,
        profile_budget=args.profile_budget,
    )
