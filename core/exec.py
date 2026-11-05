import itertools
import logging
import math
import random
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from core import op
from core.base import Operator, SubQuery, SubQueryGroup, QueryBase, TYPE_ID
from core.profiler import Profiler, get_profiler
from core.runtime_dump import dump_query_step, profile_step_metadata
from opt.logic_op import *
from opt.memo import *
from schema import get_table_schema
from config.config import HOME


logger = logging.getLogger(__name__)

class Query(QueryBase):
    def __init__(self, qid: int, db_name: str, plan_idx: int = 0):
        super().__init__(db_name, qid, plan_idx)
        self.db_name = db_name
        self.oid_map: Dict[int, Operator] = {}
        self.oid_to_op_name: Dict[int, str] = {}
        self.op_name_to_oid: Dict[str, int] = {}
        self.active_subq: Optional[SubQueryGroup] = None
        self.source_map: Dict[int, int] = {}
        self.subq_root: Optional[SubQueryGroup] = None
        self.completed_step = 0
        self.total_step = 0

        self.start_time = None
        self.end_time = None
        self.run_time = 0.0
        self.profile_time = 0.0
        self.exec_time = 0.0
        self.not_profiled = True
        self._pipeline_pending = None
        self._pipeline_profiler = None
        self._pipeline_profiled_group = None
        self._pipeline_outer_batch = False

    def get_oid(self, oid_or_op_name: Any) -> int:
        """Resolve either oid(int) or op_name(str) into an oid."""
        if isinstance(oid_or_op_name, int):
            return oid_or_op_name
        if isinstance(oid_or_op_name, str):
            if oid_or_op_name not in self.op_name_to_oid:
                raise KeyError(f"Unknown op_name '{oid_or_op_name}'")
            return self.op_name_to_oid[oid_or_op_name]
        raise TypeError(f"Expected int oid or str op_name, got {type(oid_or_op_name)}")

    def get_op_name(self, oid: int) -> Optional[str]:
        """Return op_name for oid if present in node params."""
        return self.oid_to_op_name.get(oid)

    def _build_oid_map(self, logical_plan: Plan, llm):
        """Instantiate the physical operators for the given logical plan with respective oid"""
        self.oid_map.clear()
        self.oid_to_op_name.clear()
        self.op_name_to_oid.clear()

        required_params = {
            "scan": {"table_dir"},
            "filter": {"predicate"},
            "projection": {"yields"},
            "proj": {"yields"},
            "join": {"left_on", "right_on"},
            "aggr": {"group_by", "agg_func"},
            "sort": {"order_by", "desc"},
            "sem_filter": {"key", "predicate", "llm_args"},
            "sem_proj": {"key", "new_col", "instr", "llm_args"},
            "sem_group": {"group_by", "out_col", "instr", "llm_args"},
            "sem_aggr": {"group_by", "aggr_col", "out_col", "instr", "llm_args"},
            "sem_sort": {"order_by", "instr", "desc", "llm_args"},
            "sem_join": {"left_on", "right_on", "condition", "llm_args"},
        }

        def _get_params(node: LogicalOperator) -> Dict[str, Any]:
            params = getattr(node.op_props, "params", None)
            if not isinstance(params, dict):
                raise ValueError(f"Node {node.oid} ({node.op_props.op}) must provide op_props.params as dict")

            need = required_params.get(node.op_props.op)
            if need is None:
                raise NotImplementedError(f"Unsupported operator type {node.op_props.op}")

            missing = sorted(need - set(params.keys()))
            if missing:
                raise ValueError(
                    f"Node {node.oid} ({node.op_props.op}) missing required params: {missing}"
                )
            return params

        def _register_op_name(node: LogicalOperator, params: Dict[str, Any]) -> None:
            op_name = params.get("op_name")
            existing_name = self.oid_to_op_name.get(node.oid)
            if existing_name is not None and existing_name != op_name:
                raise ValueError(f"Duplicate oid mapping conflict for {node.oid}: '{existing_name}' vs '{op_name}'")

            existing_oid = self.op_name_to_oid.get(op_name)
            if existing_oid is not None and existing_oid != node.oid:
                raise ValueError(f"Duplicate op_name mapping conflict for '{op_name}': {existing_oid} vs {node.oid}")

            self.oid_to_op_name[node.oid] = op_name
            self.op_name_to_oid[op_name] = node.oid

        def _init_op(node: LogicalOperator):
            if node is None:
                return
            params = _get_params(node)
            _register_op_name(node, params)
            if node.op_props.op == "scan":
                scan_op = op.Scan(node.oid, table_dir=HOME / params['table_dir'], plan_idx=self.plan_idx, op_name=params['op_name'], dataset=self.db_name)
                table_name = params.get('table_name')
                if table_name is None:
                    assert "table_dir" in params, f"Scan operator {node.oid} must have either table_name in params"
                scan_op.table_name = table_name
                self.oid_map[node.oid] = scan_op

            elif node.op_props.op == "filter":
                self.oid_map[node.oid] = op.Filter(node.oid, predicate=params['predicate'], key=params.get('key', None), op_name=params['op_name'], dataset=self.db_name)

            elif node.op_props.op in {"projection", "proj"}:
                self.oid_map[node.oid] = op.Projection(node.oid, columns=params['yields'], op_name=params['op_name'], dataset=self.db_name)

            elif node.op_props.op == "join":
                self.oid_map[node.oid] = op.Join(node.oid, left_on=params['left_on'], right_on=params['right_on'], op_name=params['op_name'], dataset=self.db_name)

            elif node.op_props.op == "aggr":
                self.oid_map[node.oid] = op.Aggregation(node.oid, group_by=params['group_by'], agg_func=params['agg_func'], op_name=params['op_name'], dataset=self.db_name)

            elif node.op_props.op == "sort":
                self.oid_map[node.oid] = op.Sort(node.oid, order_by=params['order_by'], desc=params['desc'], op_name=params['op_name'], dataset=self.db_name)

            elif node.op_props.op == "sem_filter":
                self.oid_map[node.oid] = op.SemanticFilter(
                    node.oid,
                    key=params['key'],
                    predicate=params['predicate'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            elif node.op_props.op == "sem_proj":
                self.oid_map[node.oid] = op.SemanticProjection(
                    node.oid,
                    key=params['key'],
                    new_col=params['new_col'],
                    instr=params['instr'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            elif node.op_props.op == "sem_group":
                self.oid_map[node.oid] = op.SemanticGroupBy(
                    node.oid,
                    group_by=params['group_by'],
                    out_col=params['out_col'],
                    instr=params['instr'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            elif node.op_props.op == "sem_aggr":
                self.oid_map[node.oid] = op.SemanticAggregation(
                    node.oid,
                    group_by=params['group_by'],
                    aggr_col=params['aggr_col'],
                    out_col=params['out_col'],
                    instr=params['instr'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            elif node.op_props.op == "sem_sort":
                self.oid_map[node.oid] = op.SemanticSort(
                    node.oid,
                    order_by=params['order_by'],
                    instr=params['instr'],
                    desc=params['desc'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            elif node.op_props.op == "sem_join":
                self.oid_map[node.oid] = op.SemanticJoin(
                    node.oid,
                    left_on=params['left_on'],
                    right_on=params['right_on'],
                    condition=params['condition'],
                    llm_client=llm,
                    llm_args=params['llm_args'],
                    op_name=params['op_name'],
                    dataset=self.db_name
                )

            else:
                raise NotImplementedError(f"Unsupported operator type {node.op_props.op}")

            self.oid_map[node.oid].plan_idx = self.plan_idx

            for child in node.children():
                _init_op(child)

        _init_op(logical_plan.root)


    def _build_lineage(self, subq_grp: SubQueryGroup):
        subq = subq_grp.subqueries[0]
        new_cols = {}

        def walk(oid: int) -> List[str]:
            if oid == -1:
                return []

            children = subq.ch_link.get(oid, [-1, -1])
            left_oid = children[0]
            right_oid = children[1] if len(children) > 1 else -1

            left_tables = walk(left_oid) if left_oid != -1 else []
            right_tables = walk(right_oid) if right_oid != -1 else []
            cur_tables = left_tables + right_tables

            cur_op = self.oid_map[oid]
            if cur_op.__type_id__ == TYPE_ID.SCAN:
                return [getattr(cur_op, 'table_name', f"scan_{oid}")]

            elif cur_op.__type_id__ == TYPE_ID.FILTER:
                for table_name in cur_tables:
                    cols = get_table_schema(self.db_name, table_name)
                    cols = cols + new_cols.get(table_name, [])
                    if any(col in cur_op.predicate for col in cols):
                        cur_op.tables = [table_name]
                        break
                return cur_tables

            elif cur_op.__type_id__ == TYPE_ID.SEM_FILTER:
                cur_op.tables = []
                for table_name in cur_tables:
                    cols = get_table_schema(self.db_name, table_name)
                    cols = cols + new_cols.get(table_name, [])
                    if cur_op.key in cols:
                        cur_op.tables = [table_name]
                        break
                if not cur_op.tables:
                    raise ValueError(
                        f"Operator {cur_op.op_name} (oid={oid}) requires column "
                        f"{cur_op.key!r}, but it is unavailable in its input tables "
                        f"{cur_tables}. Place the operator after the scan or projection "
                        "that supplies this column."
                    )
                return cur_tables

            elif cur_op.__type_id__ == TYPE_ID.SEM_PROJECTION:
                for table_name in cur_tables:
                    cols = get_table_schema(self.db_name, table_name)
                    cols = cols + new_cols.get(table_name, [])
                    if any(col in cur_op.key for col in cols):
                        cur_op.tables = [table_name]
                        break
                if cur_op.tables[0] in new_cols:
                    new_cols[cur_op.tables[0]].append(cur_op.new_col)
                else:
                    new_cols[cur_op.tables[0]] = [cur_op.new_col]
                return cur_tables

            elif cur_op.__type_id__ == TYPE_ID.JOIN or cur_op.__type_id__ == TYPE_ID.SEM_JOIN:
                # should be exactly 2 tables
                for table_name in cur_tables:
                    cols = get_table_schema(self.db_name, table_name)
                    cols = cols + new_cols.get(table_name, [])
                    if cur_op.left_on in cols:
                        cur_op.tables[0] = table_name
                    if cur_op.right_on in cols:
                        cur_op.tables[1] = table_name
                return cur_tables

            elif cur_op.__type_id__ in {TYPE_ID.AGGREGATION, TYPE_ID.SORT, TYPE_ID.SEM_GROUPBY, TYPE_ID.SEM_AGGREGATION, TYPE_ID.SEM_SORT}:
                cur_op.tables = cur_tables
                return cur_tables

            return cur_tables

        walk(subq.root)
        return


    def _get_active_subq_grp(self, root_subq: SubQueryGroup):
        """
        The subquery groups are organized in a tree structure. Scan right->left until the first node with all children _commit as true and its own _commit as false.
        If both children are already committed, the subquery can start execution. Create mock table scans for the committed results first.
        """
        if root_subq is None:
            return None

        for child_subq in reversed(root_subq.child_ptrs):
            active_subq = self._get_active_subq_grp(child_subq)
            if active_subq is not None:
                return active_subq

        if root_subq.committed():
            return None

        if all(child_subq.committed() for child_subq in root_subq.child_ptrs):
            if root_subq.executing() == False:
                root_subq.reset_source(self.source_map, self.oid_map)
            # logger.info(f"Active subquery group {root_subq.subqueries[0]}")
            return root_subq

        return None


    def init(self, logical_plan_candidates: List[Plan], llm):
        self._build_oid_map(logical_plan_candidates[0], llm)  # build the oid map based on the first candidate (the others should have exact same operators)
        self.subq_root = self.prepare_candidates(logical_plan_candidates)
        print(self.subq_root.subqueries[0].verbose(self.oid_map))
        if not self.subq_root.has_sink(self.oid_map):
            print("Here")
            sink = op.Sink()
            sink._oid = id(sink)
            sink.plan_idx = self.plan_idx
            self.oid_map[id(sink)] = sink
            for subq in self.subq_root.subqueries:
                subq._insert_sink(sink._oid)
        return


    @dump_query_step
    def step_(self, llm, profiler: Profiler, prf: bool=False, profile_enabled: bool = True,
              full_batch_execution: bool = False):
        """
        This is the main interface that executes one step (one new batch is consumed) over the active subquery.
        """
        self.active_subq = self._get_active_subq_grp(self.subq_root)

        src_ops = self.active_subq.get_source()
        tot_step = 1.0
        for src in src_ops:
            src_op = self.oid_map[src]
            tot_step *= math.ceil(src_op.tot / src_op.batch_size)
            self.total_step = tot_step

        if self.active_subq is None:
            return -1  # no active subquery, query execution is completed
        self._build_lineage(self.active_subq)

        rollback = False
        if prf and profile_enabled:
            st = perf_counter()
            logger.info(f"\n=====Profiling Step {self.completed_step}======\n")
            _, _, rollback = \
                profiler.profile(
                self.active_subq,
                self.oid_map,
                profile_enabled=profile_enabled
            )
            self.profile_time += perf_counter() - st
            logger.critical(f"Profiling used: {perf_counter() - st:.4f} seconds")
            self.not_profiled = False

        logger.info(f"\n======Executing Step {self.completed_step}======\n")
        st = perf_counter()
        subq = self.active_subq.subqueries[self.active_subq.best_id]
        logger.info(f"Executing subquery {subq.verbose(self.oid_map)}")
        subq._use_llm(self.oid_map, llm)
        subq._profile_enabled = profile_enabled
        subq.full_batch_execution = full_batch_execution
        subq.capture_execution_feedback = profiler is not None and profile_enabled
        feedback_context = (profiler._execution_inputs(subq, self.oid_map)
                            if profiler is not None else None)
        out = subq.commit(self.oid_map, rollback=rollback)
        if subq.capture_execution_feedback:
            profiler.ingest_execution(self.active_subq, subq, self.oid_map,
                                      subq._last_execution_trace, feedback_context)
        exec_time = perf_counter() - st
        self.exec_time += exec_time
        logger.critical(f"Execution used: {exec_time:.4f} seconds")

        if out is not None:
            sink_op = op.Source(data=out.data, tables=out.tables)
            sink_op._oid = max(self.oid_map) + 1
            logger.critical(f"Created sink operator {sink_op._oid} with output batch of size {len(out.data)}")
            self.oid_map[sink_op._oid] = sink_op
            original_root = self.active_subq.root_oid()
            self.source_map[original_root] = sink_op._oid

        # synchronize exec state for all subqueries in the group
        for s in self.active_subq.subqueries:
            s._exec_state = subq._exec_state
            s.state_backup = subq.state_backup

        step_time = perf_counter() - st
        self.completed_step += 1

        if out is not None and self.active_subq.subqueries[0].root == self.subq_root.subqueries[0].root:
            logger.info(f"Query {self.qid} completed with output batch of size {len(out.data)}")
            step_time = -1
        return step_time

    def get_progress(self):
        return self.completed_step / self.total_step if self.total_step > 0 else 0.0

    def _profile_ahead(self, profiler, executor):
        """Profile the next input on private operators while this step executes."""
        live = self.active_subq.subqueries[self.active_subq.best_id]
        current = profiler._execution_inputs(live, self.oid_map)
        if current['eof']:
            self._pipeline_pending = None
            return
        # Keep model clients shared, but give profiling its own operator state.
        clients = [profiler.model]
        clients.extend(getattr(operator, 'llm_client', None)
                       for operator in self.oid_map.values())
        memo = {id(model): model for model in clients if model is not None}
        for operator in self.oid_map.values():
            if operator.__type_id__ == TYPE_ID.SCAN:
                memo[id(operator.table)] = operator.table
            elif operator.__type_id__ == TYPE_ID.SOURCE:
                memo[id(operator.buffer)] = operator.buffer
        group, operators = deepcopy((self.active_subq, self.oid_map), memo)
        for oid, cursor in current['post_cursors'].items():
            operators[oid].cursor = cursor
        for oid, cursor in current.get('post_join_cursors', {}).items():
            operators[oid].right_cursor = cursor
        worker_profiler = Profiler(profiler.model, profiler.profile_budget)
        worker_profiler.step_counter = profiler.step_counter
        worker_profiler.call_overhead_cost = profiler.call_overhead_cost
        worker_profiler._call_fit = profiler._call_fit
        worker_profiler._observations.update({
            oid: history.copy() for oid, history in profiler._observations.items()})
        next_step = self.completed_step + 2
        group._needs_reprofile = True

        def prepare():
            started = perf_counter()
            with profile_step_metadata(self, next_step):
                result = worker_profiler.profile(group, operators)
            return group, operators, worker_profiler, result, perf_counter() - started

        self._pipeline_pending = (self.active_subq, executor.submit(prepare))

    def _accept_profile_ahead(self, profiler):
        pending = self._pipeline_pending
        self._pipeline_pending = None
        if pending is None or pending[0] is not self.active_subq:
            return False
        try:
            group, operators, prepared, _, elapsed = pending[1].result()
        except Exception:
            logger.exception('Speculative profile failed; profiling the live input')
            return False
        self.profile_time += elapsed
        live = self.active_subq.subqueries[self.active_subq.best_id]
        actual = profiler._execution_inputs(live, self.oid_map)
        predicted = prepared._sampling_context

        keys = ('cursors', 'eof', 'leaves', 'joins', 'completed')
        if any(actual[key] != predicted[key] for key in keys):
            return False
        for oid, copied in operators.items():
            sample = getattr(copied, '_profile_sample', None)
            if sample is not None and oid in self.oid_map:
                self.oid_map[oid]._profile_sample = sample
        self.active_subq.best_id = group.best_id
        self.active_subq._profile_cost_context = group._profile_cost_context
        self.active_subq._last_complete_sample_step = group._last_complete_sample_step
        self.active_subq._predicted_step_cost = group._predicted_step_cost
        self.active_subq._predicted_cost_supported = group._predicted_cost_supported
        self.active_subq._dump_sample_step = group._dump_sample_step
        self.active_subq._dump_sample_query_step = getattr(group, '_dump_sample_query_step', None)
        self.active_subq._execution_feedback_ready = False
        selected = group.subqueries[group.best_id]
        self.active_subq.subqueries[group.best_id]._dump_prediction_link = \
            getattr(selected, '_dump_prediction_link', {})
        profiler.step_counter = prepared.step_counter
        return True

    @dump_query_step
    def step_pipelined_(self, llm, profiler: Profiler, executor, prf=True,
                        profile_enabled=True, full_batch_execution=False,
                        profile_interval=1):
        """Execute the last profiled plan, sampling ahead on scheduled steps."""
        if profile_interval < 1:
            raise ValueError('profile_interval must be at least 1')
        self.active_subq = self._get_active_subq_grp(self.subq_root)
        if self.active_subq is None:
            return -1
        self._build_lineage(self.active_subq)
        sources = self.active_subq.get_source()
        total = 1.0
        for oid in sources:
            source = self.oid_map[oid]
            total *= math.ceil(source.tot / source.batch_size)
        self.total_step = total

        if self._pipeline_profiler is None:
            self._pipeline_profiler = Profiler(profiler.model, profiler.profile_budget)
        pipeline_profiler = self._pipeline_profiler
        rollback = False
        profile_reasons = []
        # A step that sent no prompts leaves nothing for another plan to save, so scheduled profiles wait until execution does work again.
        idle = pipeline_profiler.last_step_cost == 0
        if self.completed_step % profile_interval == 0 and not (
                idle and self._pipeline_pending is None):
            profile_reasons.append('interval')
        if self._pipeline_outer_batch:
            profile_reasons.append('outer batch')
        self._pipeline_outer_batch = False
        if self._pipeline_profiled_group is not self.active_subq:
            profile_reasons.append('new group')
        if getattr(self.active_subq, '_needs_reprofile', False):
            profile_reasons.append('execution drift')
        profile_due = prf and profile_enabled and bool(profile_reasons)
        if profile_due:
            started = perf_counter()
            accepted = self._accept_profile_ahead(pipeline_profiler)
            if not accepted:
                self.active_subq._needs_reprofile = True
                _, _, rollback = pipeline_profiler.profile(self.active_subq, self.oid_map)
                self.profile_time += perf_counter() - started
            self._pipeline_profiled_group = self.active_subq
            self.not_profiled = False

        subq = self.active_subq.subqueries[self.active_subq.best_id]
        logger.info("Query %d pipelined step %d selected candidate %d (%s):\n%s",
                    self.qid, self.completed_step + 1, self.active_subq.best_id,
                    'profiled: ' + ', '.join(profile_reasons) if profile_due else 'reused',
                    subq.verbose(self.oid_map))
        subq._use_llm(self.oid_map, llm)
        subq._profile_enabled = profile_enabled
        subq.full_batch_execution = full_batch_execution
        subq.capture_execution_feedback = True
        profile_ahead = prf and profile_enabled and not idle and (
            (self.completed_step + 1) % profile_interval == 0 or
            pipeline_profiler.outer_batch_follows(subq, self.oid_map))
        if profile_ahead:
            self._pipeline_outer_batch = (self.completed_step + 1) % profile_interval != 0
            self._profile_ahead(pipeline_profiler, executor)

        started = perf_counter()
        out = subq.commit(self.oid_map, rollback=rollback)

        if pipeline_profiler.observe_execution(subq._last_execution_trace):
            self.active_subq._needs_reprofile = True
        elapsed = perf_counter() - started
        self.exec_time += elapsed
        if out is not None:
            sink = op.Source(data=out.data, tables=out.tables)
            sink._oid = max(self.oid_map) + 1
            self.oid_map[sink._oid] = sink
            self.source_map[self.active_subq.root_oid()] = sink._oid
        for candidate in self.active_subq.subqueries:
            candidate._exec_state = subq._exec_state
            candidate.state_backup = subq.state_backup
        self.completed_step += 1
        if out is not None and self.active_subq.subqueries[0].root == self.subq_root.subqueries[0].root:
            return -1
        return perf_counter() - started


class Scheduler:
    POLICIES = ('srtf', 'fcfs')
    def __init__(self, profile_enabled: bool = True, full_batch_execution: bool = False,
                 fully_pipelined: bool = False, pipeline_profile_interval: int = 1,
                 srtf_alpha: float = 0.1, srtf_smoothing: float = 0.2, policy: str = 'srtf'):
        if pipeline_profile_interval < 1:
            raise ValueError('pipeline_profile_interval must be at least 1')
        if policy not in self.POLICIES:
            raise ValueError(f'Unknown scheduling policy {policy!r}')
        if not math.isfinite(srtf_alpha) or srtf_alpha < 0:
            raise ValueError('srtf_alpha must be finite and nonnegative')
        if not math.isfinite(srtf_smoothing) or not 0 < srtf_smoothing <= 1:
            raise ValueError('srtf_smoothing must be in (0, 1]')
        self.queue: List[int] = []  # qids, in (re)queue order for SRTF ties
        self.qid_map: Dict[int, Query] = {}
        self.est_rem_time = {qid: float('inf') for qid in self.queue}
        self._cv = threading.Condition()
        self._stop = False
        self.policy = policy
        self.srtf_alpha = srtf_alpha
        self.srtf_smoothing = srtf_smoothing
        # Per-query service time and smoothed estimate, as in sim.load.Job.
        self._srtf_state: Dict[int, Dict[str, Any]] = {}
        self.profile_enabled = profile_enabled
        self.full_batch_execution = full_batch_execution
        self.fully_pipelined = fully_pipelined
        self.pipeline_profile_interval = pipeline_profile_interval
        self._profile_executor = ThreadPoolExecutor(max_workers=4) if fully_pipelined else None
        self.step_counter = 0

        try:
            self.profiler = get_profiler()
        except Exception as e:
            logger.warning(f"Failed to initialize profiler, profiling will be disabled: {e}")
            self.profiler = None

    def add_query(self, query: Query):
        with self._cv:
            if query.qid in self.qid_map:
                self.qid_map[query.qid] = query
            else:
                self.queue.append(query.qid)
                self.qid_map[query.qid] = query
                self.est_rem_time[query.qid] = float('inf')
                self._srtf_state[query.qid] = dict(served=0.0, smoothed=None, step=None,
                                                   served_at_estimate=0.0)
                query.start_time = perf_counter()
            self._cv.notify()

    def stop(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    def sched(self, cur_id):
        # currently this is a round-robin
        with self._cv:
            if not self.queue:
                return None
            return (cur_id + 1) % len(self.queue)

    def sched_srtf(self, last_step_cost, last_est_prog):
        if not self.queue:
            return None

        for qid in self.queue:
            tot_prog = self.qid_map[qid].get_progress()

            last_prog = last_est_prog.get(qid, 0.0)
            last_cost = last_step_cost.get(qid, float('inf'))
            if tot_prog != last_prog:
                self.est_rem_time[qid] = (last_cost / (tot_prog - last_prog)) * (1-tot_prog)

        srtf = self._select()
        logger.critical(f"Scheduling query {srtf} with estimated remaining time {self.est_rem_time.get(srtf, float('inf')):.4f} seconds")
        return srtf

    def _first(self):
        """Query to run when none is current."""
        return self.queue[0]

    def _select(self):
        """Query to run after a step, once remaining-time estimates are updated."""
        # min keeps the first of equal keys, so ties go to the earliest (re)queued query.
        if self.policy == 'fcfs':
            # As sim.load: the earliest arrival runs until it completes.
            return min(self.queue, key=lambda qid: self.qid_map[qid].start_time)
        return min(self.queue, key=self._srtf_priority)

    def _on_step(self, qid, step_time, done):
        """Called after every step of a known query, before it is requeued or removed."""

    def _srtf_priority(self, qid):
        state = self._srtf_state[qid]
        raw = self.est_rem_time.get(qid, float('inf'))
        if not math.isfinite(raw):
            raw = 0.0
        if self.srtf_smoothing == 1:
            remaining = raw
        else:
            step = self.qid_map[qid].completed_step
            if state['step'] != step:
                observed = max(0.0, raw)
                if state['smoothed'] is None:
                    if observed > 0:
                        state['smoothed'] = observed
                else:
                    consumed = max(0.0, state['served'] - state['served_at_estimate'])
                    residual = max(0.0, state['smoothed'] - consumed)
                    weight = self.srtf_smoothing
                    state['smoothed'] = (1 - weight) * residual + weight * observed
                state['step'] = step
                state['served_at_estimate'] = state['served']
            remaining = state['smoothed'] if state['smoothed'] is not None else 0.0
        arrival = self.qid_map[qid].start_time
        return remaining + self.srtf_alpha * (arrival + state['served'])


    def run(self, llm, serve: bool = False, prf: bool = True):
        qid = None
        last_step_cost = {}
        last_est_prog = {}

        while True:
            with self._cv:
                while not self.queue and not self._stop:
                    if not serve:
                        if self._profile_executor is not None:
                            self._profile_executor.shutdown(wait=True)
                        return
                    self._cv.wait()

                if self._stop:
                    if self._profile_executor is not None:
                        self._profile_executor.shutdown(wait=True)
                    return

                if qid not in self.qid_map:
                    qid = self._first()

                query = self.qid_map[qid]
                last_step_cost.setdefault(qid, float('inf'))
                last_est_prog.setdefault(qid, 0.0)

            # if self.step_counter % 100 == 0 or query.not_profiled:
            #     cost = query.step_(llm=llm, profiler=self.profiler, prf=prf, profile_enabled=self.profile_enabled)
            # else:
            #     cost = query.step_(llm=llm, profiler=self.profiler, prf=False, profile_enabled=False)

            step_started = perf_counter()
            if self.fully_pipelined and prf and self.profile_enabled and self.profiler is not None:
                cost = query.step_pipelined_(llm=llm, profiler=self.profiler,
                                             executor=self._profile_executor,
                                             full_batch_execution=self.full_batch_execution,
                                             profile_interval=self.pipeline_profile_interval)
            else:
                cost = query.step_(llm=llm, profiler=self.profiler, prf=prf,
                                   profile_enabled=self.profile_enabled,
                                   full_batch_execution=self.full_batch_execution)
            step_time = perf_counter() - step_started
            self.step_counter += 1

            with self._cv:
                if qid in self.qid_map:
                    self._on_step(qid, step_time, cost < 0)
                    if cost < 0:  # query completed:
                        self.qid_map[qid].end_time = perf_counter()
                        logger.critical(f"Query {qid} completed with total run time {self.qid_map[qid].end_time - self.qid_map[qid].start_time:.4f} s, profile time {self.qid_map[qid].profile_time:.4f} s, exec time {self.qid_map[qid].exec_time:.4f} s")
                        self.queue.remove(qid)
                        del self.qid_map[qid]
                        self.est_rem_time.pop(qid, None)
                        self._srtf_state.pop(qid, None)
                        qid = None
                    else:
                        last_step_cost[qid] = cost
                        self._srtf_state[qid]['served'] += step_time
                        # Requeue behind queries that arrived during the step.
                        self.queue.remove(qid)
                        self.queue.append(qid)

                if not self.queue:
                    if not serve:
                        return
                    continue

                next_qid = self.sched_srtf(last_step_cost, last_est_prog)
                if next_qid is None:
                    qid = None
                    continue
                qid = next_qid
                last_est_prog[qid] = self.qid_map[qid].get_progress()

    def serve(self, llm, prf: bool = True):
        self.run(llm=llm, serve=True, prf=prf)


class StepwiseScheduler(Scheduler):

    POLICIES = ('mlfq', 'pdb')

    def __init__(self, policy: str = 'mlfq', mlfq_levels: int = 4,
                 mlfq_base_quantum: float = 1.0, mlfq_boost_interval: float = 60.0,
                 pdb_batch_size: int = 4, pdb_starvation_threshold: float = 0.75,
                 pdb_admission_interval: float = 1.0, pdb_high_fraction: float = 0.2,
                 pdb_seed: int = 0, **kwargs):
        """Remaining keyword arguments are Scheduler's (profiling, pipelining, ...)."""
        if not isinstance(mlfq_levels, int) or mlfq_levels < 1:
            raise ValueError('mlfq_levels must be a positive integer')
        for value, name in ((mlfq_base_quantum, 'mlfq_base_quantum'),
                            (mlfq_boost_interval, 'mlfq_boost_interval'),
                            (pdb_admission_interval, 'pdb_admission_interval')):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not isinstance(pdb_batch_size, int) or pdb_batch_size < 1:
            raise ValueError('pdb_batch_size must be a positive integer')
        for value, name in ((pdb_starvation_threshold, 'pdb_starvation_threshold'),
                            (pdb_high_fraction, 'pdb_high_fraction')):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1]')
        super().__init__(policy=policy, **kwargs)
        self._policy_state: Dict[int, Dict[str, Any]] = {}
        # MLFQ
        self._quanta = [mlfq_base_quantum * 2 ** level for level in range(mlfq_levels)]
        self.mlfq_boost_interval = mlfq_boost_interval
        self._next_boost = None
        self._order = itertools.count()
        # PreemptDB
        self.pdb_batch_size = pdb_batch_size
        self.pdb_starvation_threshold = pdb_starvation_threshold
        self.pdb_admission_interval = pdb_admission_interval
        self.pdb_high_fraction = pdb_high_fraction
        self._pdb_rng = random.Random(pdb_seed)
        self._high_pending: List[int] = []
        self._high_batch: List[int] = []
        self._low_ready: List[int] = []  # FIFO; a running low query stays at the front
        self._paused = None              # low query preempted by an admitted high batch
        self._active = None              # query that just ran a step and is unfinished
        self._next_admission = math.inf

    def add_query(self, query, high: Optional[bool] = None):
        """`high` tags a PreemptDB priority; None draws it with pdb_high_fraction."""
        with self._cv:
            new = query.qid not in self.qid_map
            super().add_query(query)
            if new:
                self._register(query.qid, high, query.start_time)

    # ---- Scheduler hooks ----

    def _first(self):
        return self._pick()

    def _select(self):
        return self._pick()

    def _on_step(self, qid, step_time, done):
        state = self._policy_state[qid]
        if self.policy == 'mlfq':
            if not done:
                state['used'] += step_time
                if state['used'] >= self._quanta[state['level']] - 1e-9:
                    state['level'] = min(state['level'] + 1, len(self._quanta) - 1)
                    state['used'] = 0.0
                state['order'] = next(self._order)  # requeue at the back of its level
        else:
            if state['high'] and self._paused is not None:
                self._policy_state[self._paused]['high_service'] += step_time
            if done:
                if state['high']:
                    self._high_batch.remove(qid)
                    # End the batch early once the paused low query is starving.
                    now = perf_counter()
                    if (self._paused is not None and
                            self._starvation(self._paused, now) >= self.pdb_starvation_threshold):
                        self._high_pending[0:0] = self._high_batch
                        self._high_batch.clear()
                else:
                    self._low_ready.remove(qid)
            self._active = None if done else qid
        if done:
            del self._policy_state[qid]

    # ---- Policy internals ----

    def _register(self, qid, high, now):
        if self.policy == 'mlfq':
            self._policy_state[qid] = dict(level=0, used=0.0, order=next(self._order))
            if self._next_boost is None:
                self._next_boost = now + self.mlfq_boost_interval
            return
        if high is None:
            high = self._pdb_rng.random() < self.pdb_high_fraction
        self._policy_state[qid] = dict(high=bool(high), start=None, high_service=0.0)
        if high:
            self._high_pending.append(qid)
            if math.isinf(self._next_admission):
                self._next_admission = self._next_admission_tick(now)
        else:
            self._low_ready.append(qid)

    def _next_admission_tick(self, now):
        interval = self.pdb_admission_interval
        return (math.floor(now / interval) + 1) * interval

    def _starvation(self, qid, now):
        state = self._policy_state[qid]
        if state['start'] is None or now <= state['start']:
            return 0.0
        return state['high_service'] / (now - state['start'])

    def _pdb_admit(self, now):
        """Admit one bounded high batch, pausing the low query that just ran."""
        if not self._high_pending or self._high_batch:
            return
        idle = self._paused is None and not self._low_ready
        if not idle and now < self._next_admission:
            return
        active = self._active
        low = active if active is not None else self._paused
        if idle or low is None or self._starvation(low, now) < self.pdb_starvation_threshold:
            count = min(self.pdb_batch_size, len(self._high_pending))
            self._high_batch.extend(self._high_pending[:count])
            del self._high_pending[:count]
            if active is not None:
                self._low_ready.remove(active)
                self._paused = active
        if now >= self._next_admission:
            self._next_admission = (self._next_admission_tick(now) if self._high_pending
                                    else math.inf)

    def _pick(self):
        now = perf_counter()
        if self.policy == 'mlfq':
            if self._next_boost is not None and now >= self._next_boost:
                for state in self._policy_state.values():
                    state['level'], state['used'] = 0, 0.0
                while self._next_boost <= now:
                    self._next_boost += self.mlfq_boost_interval
            return min(self.queue, key=lambda qid: (self._policy_state[qid]['level'],
                                                    self._policy_state[qid]['order']))
        self._pdb_admit(now)
        if self._high_batch:
            return self._high_batch[0]
        if self._paused is not None:
            self._low_ready.insert(0, self._paused)
            self._paused = None
        qid = self._low_ready[0]
        state = self._policy_state[qid]
        if state['start'] is None:
            state['start'], state['high_service'] = now, 0.0
        return qid
