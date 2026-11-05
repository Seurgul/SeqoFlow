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
from config.config import HOME, prepare_plan_config
from config.randomize import set_use_config
from frontend.build import create_workspace
from core.exec import Query, Scheduler
from core.llm import VLLMWrapper
from core.profiler import Profiler, get_profiler, init_profiler


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan enumeration script")
    parser.add_argument("--dataset", type=str, required=True, choices=['bio', 'arxiv', 'legal', 'patent'])
    parser.add_argument("--do_cfg", type=int, choices=[0, 1], default=1, help="enable randomized operator configs for the entire run (0 disables, 1 enables)")
    parser.add_argument("--plan_idx", type=int, required=True)
    args = parser.parse_args()
    set_use_config(args.do_cfg == 1)

    plan_name, path = create_workspace(f"{HOME}/data/{args.dataset}/t{args.plan_idx}.json")

    st = time.perf_counter()
    plan = Plan(load_plan(path))
    # plan.assign_oid()
    plan.populate_props()
    plan.pushup(plan.root)
    dump_plan(plan.root, f"{HOME}/plans/{plan_name}/{plan_name}_pu.json")
    plan.pushdown()
    ed = time.perf_counter()
    memo = get_memoizer()
    logger.critical(f"Plan enumeration for t{args.plan_idx} took {ed - st:.4f} seconds, {len(memo._memo)} plans enumerated.")

    for i, cand in enumerate(memo._memo):
        cand.populate_props()
        dump_plan(cand.root, f"{HOME}/plans/{plan_name}/{plan_name}_{i}.json")
    logger.critical(f"Wrote {len(memo._memo)} candidates to {HOME}/plans/{plan_name}/")
    memo.clear()
