import logging
from typing import Protocol, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum, auto

import pandas as pd

from core.batch import Batch
from core.execution_mode import set_full_batch_execution, reset_full_batch_execution
from core.runtime_dump import dump_execution
from opt.logic_op import Plan, LogicalOperator


logger = logging.getLogger(__name__)

class TYPE_ID(Enum):
    SCAN = auto()
    FILTER = auto()
    PROJECTION = auto()
    JOIN = auto()
    AGGREGATION = auto()
    SORT = auto()
    SEM_FILTER = auto()
    SEM_PROJECTION = auto()
    SEM_GROUPBY = auto()
    SEM_AGGREGATION = auto()
    SEM_SORT = auto()
    SEM_JOIN = auto()
    SINK = auto()
    SOURCE = auto()

class Operator(Protocol):
    dataset: str
    plan_idx: int

    @property
    def oid(self) -> int:
        pass

    def process(self, payload: Batch = None, is_profile=False) -> Optional[Batch]:
        """A pull-based implementation"""
        pass

    def _get_profile(self):
        return None


@dataclass(eq=False)
class SubQuery:
    root: int = None
    ch_link: Dict[int, List[int]] = field(default_factory=dict)  # parent oid -> list of child oids
    parent_link: Dict[int, int] = field(default_factory=dict)  # child oid -> parent oid
    parent_ptr: Optional["SubQuery"] = None
    child_ptrs: List["SubQuery"] = field(default_factory=list)
    _exec_state: Dict = field(default_factory=dict, repr=False)
    _sched_state: str = "READY"
    state_backup: Dict = field(default_factory=dict, repr=False)
    _profile_enabled: bool = True
    full_batch_execution: bool = False
    capture_execution_feedback: bool = False

    def add_link(self, p_oid, ch_oid, br=None):
        if p_oid not in self.ch_link:
            self.ch_link[p_oid] = [-1, -1]
        if ch_oid is None:
            return
        if br == "left":
            self.ch_link[p_oid][0] = ch_oid
        elif br == "right":
            self.ch_link[p_oid][1] = ch_oid
        else:
            self.ch_link[p_oid][0] = ch_oid
        self.parent_link[ch_oid] = p_oid

    def _get_ops(self, oid_map: Dict[int, Operator]):
        """return all physical operators as a list"""
        operators = []
        def _dfs(oid):
            if oid == -1: return
            operators.append(oid_map[oid])
            for child in self.ch_link.get(oid, [-1]):
                _dfs(child)
        _dfs(self.root)
        return operators

    def _get_leaf_ops(self):
        """return all leaf operators as a list"""
        leaf_ops = []
        def _dfs(oid):
            if oid == -1: return
            children = self.ch_link.get(oid, [-1, -1])
            if children == [-1, -1]:
                leaf_ops.append(oid)
            for child in children:
                _dfs(child)
        _dfs(self.root)
        return leaf_ops

    def _get_join_ops(self, oid_map: Dict[int, Operator]):
        """return all join operators as a list"""
        join_ops = []
        def _dfs(oid):
            if oid == -1: return
            op = oid_map[oid]
            if op.__type_id__ in {TYPE_ID.SEM_JOIN}:
                join_ops.append(op)
            for child in self.ch_link.get(oid, [-1, -1]):
                _dfs(child)
        _dfs(self.root)
        return join_ops

    def _use_llm(self, oid_map: Dict[int, Operator], llm):
        """Set all operators to use an LLM"""
        ops = self._get_ops(oid_map)
        for op in ops:
            if hasattr(op, "llm_client"):
                op.llm_client = llm

    def __str__(self):
        output = []

        def _print(node, indent=0):
            if node == -1:
                return
            output.append(" " * indent + f"└── {node}")
            for child in self.ch_link.get(node, [-1]):
                _print(child, indent + 2)

        _print(self.root)
        return "\n".join(output)

    def verbose(self, oid_map: Dict[int, Operator]):
        output = []

        def _print(node, indent=0):
            if node == -1:
                return
            op = oid_map[node]
            output.append(" " * indent + f"└── {op.op_name}")
            for child in self.ch_link.get(node, [-1]):
                _print(child, indent + 2)

        _print(self.root)
        return "\n".join(output)

    def _deactivate_ops(self, oid_map: Dict[int, Operator]):
        ops = self._get_ops(oid_map)
        for op in ops:
            if hasattr(op, "active"):
                op.active = False

    def _insert_sink(self, sink_id: int):
        """Reset root op to sink"""
        old_root = self.root
        self.root = sink_id
        self.add_link(sink_id, old_root)

    def _reset_source(self, original, new_source_id):
        """Replace all original in ch_link and parent_link with new_source_id"""
        logger.critical(f"Resetting source from {original} to {new_source_id} in subquery with root {self.root}")
        for p_oid, children in self.ch_link.items():
            self.ch_link[p_oid] = [new_source_id if child == original else child for child in children]
        for ch_oid, p_oid in self.parent_link.items():
            if p_oid == original:
                self.parent_link[ch_oid] = new_source_id

    @dump_execution
    def execute(self, op_oid_map: Dict[int, Operator], is_profile=False, rollback=False):
        """
        Advance the subquery by exactly one step in a pull-based model. On the first call every leaf produces one batch and the results flow up to the root.  Subsequent calls advance the *rightmost* data source by one batch.  For each Join the right branch is materialised on the first pass and replayed for every new left-branch batch, implementing a nested-loop join where the rightmost input cycles fastest.

        Returns the Batch emitted by the root operator, or an EOF Batch once
        all input has been consumed.
        """
        _EMPTY = Batch(idx=[-1], data=pd.DataFrame())          # reusable dummy for replay triggers
        _EOF = Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)

        def _stream_ckpt(root_oid: int):
            if self.ch_link[root_oid][0] == -1:
                return []
            stream = [op_oid_map[root_oid]]
            cur = root_oid
            while True:
                ch = self.ch_link.get(cur, [-1])
                if len(ch) == 2 and ch[0] != -1 and ch[1] != -1:
                    break
                if ch[0] == -1:
                    break
                stream.append(op_oid_map[ch[0]])
                cur = ch[0]
            stream.pop()
            print(f"Checkpoint stream for root {root_oid}: {[op.oid for op in stream]}")
            return stream

        def _pull(oid: int):
            """Recursively pull one output batch from the subtree rooted at oid."""
            if oid == -1:
                return _EOF

            op_inst = op_oid_map[oid]
            children = self.ch_link.get(oid, [-1, -1])
            left_oid = children[0]
            right_oid = children[1] if len(children) > 1 else -1

            # Leaf
            if left_oid == -1:
                logger.info(f"Pulling leaf operator {op_oid_map[oid].table_name}")
                batch = op_inst.process(is_profile=is_profile)
                return batch

            # Unary
            if right_oid == -1:
                logger.info(f"Pulling {op_oid_map[oid].__type_id__} {op_oid_map[oid].tables if hasattr(op_oid_map[oid], 'tables') else ''}")
                child_batch = _pull(left_oid)
                return op_inst.process(child_batch, is_profile=is_profile)

            # Binary (Join)
            j = op_inst
            state = self._exec_state.setdefault(oid, {"phase": "init"})  # init, materializing, replaying

            # if profile, backup first.
            if self._profile_enabled:
                if is_profile:
                    j._backup()
                elif rollback:
                    j._restore()

            # INIT pull first left batch, arm the join, fall through to MATERIALIZING
            left_stream = _stream_ckpt(left_oid)
            right_stream = _stream_ckpt(right_oid)
            if state['phase'] == 'init':
                logger.info(f"Initializing Join {oid} with first left batch")
                left_batch = _pull(left_oid)
                if left_batch.is_eof:
                    return left_batch

                j.process(left_batch, 'left', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))
                state['phase'] = 'materializing'

            # MATERIALIZING pull right batches one at a time; emit joined rows.
            # When the right side is exhausted, fetch a fresh left batch and start replaying the buffered right side.
            if state['phase'] == 'materializing':
                logger.info(f"Materializing right side of Join {oid}")
                right_batch = _pull(right_oid)
                result = j.process(right_batch, 'right', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))    # None when right_eof just set
                if j.right_eof:
                    # Right branch fully materialised, right_buffer[-1] is the EOF
                    left_batch = _pull(left_oid)
                    if left_batch.is_eof:
                        return left_batch
                    logger.info(f"Join {oid} finished materializing right batches, starting replay with new left batch")
                    j.process(left_batch, 'left', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))
                    logger.info(f"Replaying Join {oid} with right batch")
                    result = j.process(_EMPTY, 'right', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))     # replay right_buffer[0]
                    state['phase'] = 'replaying'

                return result

            # REPLAYING cursor advances through right_buffer[0, ..., N-1].
            # When cursor reaches N (== len-1) the last real batch was already emitted on the previous call; reset the cursor and get a new left.
            if j.right_cursor >= len(j.right_buffer) - 1:
                logger.info(f"Join {oid} finished replaying right batches, fetching new left batch")
                j.process(_EOF, 'right', is_profile=is_profile)     # resets right_cursor to 0
                left_batch = _pull(left_oid)
                if left_batch.is_eof: return left_batch
                j.process(left_batch, 'left', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))

            logger.info(f"Replaying Join {oid} with right batch")
            return j.process(_EMPTY, 'right', is_profile=is_profile, upstream_ckpt=(left_stream, right_stream))       # emit next replayed row

        token = set_full_batch_execution(self.full_batch_execution and not is_profile)
        try:
            return _pull(self.root)
        finally:
            reset_full_batch_execution(token)


    def commit(self, op_oid_map: Dict[int, Operator], is_profile=False, rollback=False):
        """
        A wrapper for execute()
        """
        logger.info(f"rollback {rollback} | is_profile {is_profile}")
        self._sched_state = "EXECUTE"
        if self._profile_enabled:
            if is_profile:
                self.state_backup = self._exec_state.copy()  # backup state for potential re-execution during profiling
            elif rollback:
                self._exec_state = self.state_backup.copy()  # restore state after profiling

        output_batch = self.execute(op_oid_map, is_profile=is_profile, rollback=rollback)
        self._deactivate_ops(op_oid_map)

        if self._profile_enabled:
            if is_profile is False:
                self.state_backup = self._exec_state.copy()  # backup state after real execution
        if output_batch is not None:
            self._sched_state = "COMMIT"
            logger.info(f"SubQuery with root {self.root} committed with output batch of size {len(output_batch.data)}")
        return output_batch


@dataclass(eq=False)
class SubQueryGroup:
    subqueries: List[SubQuery] = field(default_factory=list)
    parent_ptr: Optional["SubQueryGroup"] = None
    child_ptrs: List["SubQueryGroup"] = field(default_factory=list)
    best_id: int = 0

    def has(self, subq: SubQuery) -> bool:
        return subq in self.subqueries

    def committed(self) -> bool:
        return any(sq._sched_state == "COMMIT" for sq in self.subqueries)

    def executing(self) -> bool:
        return any(sq._sched_state == "EXECUTE" for sq in self.subqueries)

    def has_sink(self, oid_map) -> bool:
        root = oid_map[self.subqueries[0].root]
        return root.__type_id__ in {TYPE_ID.AGGREGATION, TYPE_ID.SORT, TYPE_ID.SEM_AGGREGATION, TYPE_ID.SEM_SORT, TYPE_ID.SEM_GROUPBY, TYPE_ID.SINK}

    def root_oid(self) -> int:
        return self.subqueries[0].root

    def reset_source(self, source_map: Dict[int, int], oid_map: Dict[int, Operator]):
        for sq in self.subqueries:
            leaf_op_oids = sq._get_leaf_ops()
            for oid in leaf_op_oids:
                if oid_map[oid].__type_id__ == TYPE_ID.SCAN:
                    continue
                src_oid = source_map.get(oid, None)
                assert src_oid is not None, f"Source mapping missing for leaf op {oid}"
                assert src_oid != oid, f"Source mapping for leaf op {oid} points to itself, cycle detected!"
                sq._reset_source(oid, src_oid)

        sq = self.subqueries[0]
        print(sq.ch_link)
        join_ops = sq._get_join_ops(oid_map)
        leaf_ops = sq._get_leaf_ops()
        tot_batches = 1.0
        for leaf in leaf_ops:
            tot_batches *= oid_map[leaf].tot / oid_map[leaf].batch_size
        for join in join_ops:
            join.total_batches = tot_batches

    def get_source(self):
        return self.subqueries[0]._get_leaf_ops()


class QueryBase:
    def __init__(self, dataset: str, qid: int, plan_idx: int = 0):
        self.qid = qid
        self.plan_idx = plan_idx
        self.dataset = dataset

    def build_plan(self, logical_plan: Plan):
        subquery = SubQuery(root=logical_plan.root.oid)

        def _build(node: LogicalOperator, subquery):
            if node is None:
                return
            if node.oid != logical_plan.root.oid and node.op_props.op in {"sem_aggr", "sem_sort", "aggr", "sort", "sem_group"}:
                # print(subquery.__str__())
                new_subquery = SubQuery(root=node.oid)
                new_subquery.parent_ptr = subquery
                subquery.child_ptrs.append(new_subquery)
                subquery = new_subquery
            if len(node.children()) == 2:
                for i, child in enumerate(node.children()):
                    subquery.add_link(node.oid, child.oid, "left" if i == 0 else "right")
                    _build(child, subquery)
            elif len(node.children()) == 1:
                subquery.add_link(node.oid, node.children()[0].oid)
                _build(node.children()[0], subquery)
            else:
                subquery.add_link(node.oid, None)

        _build(logical_plan.root, subquery)
        return subquery

    def get_subq(self, subquery: SubQuery):
        """
        Return the subqueries in topological order (from leaf to root).
        """
        subqueries = []

        def _dfs(sq: SubQuery):
            for child in sq.child_ptrs:
                _dfs(child)
            subqueries.append(sq)

        _dfs(subquery)
        return subqueries

    def prepare_candidates(self, logical_plan_candidates: List[Plan]):
        """All logical plans should have the same topology in subquery structuring. This function will get all peers in the same slot and group them"""
        subqueries = []
        for logical_plan in logical_plan_candidates:
            subqueries.append([subq for subq in self.get_subq(self.build_plan(logical_plan))])

        assert len(set(len(s) for s in subqueries)) == 1, "All plans should have the same number of subqueries"
        num_grps = len(subqueries[0])
        grps = [SubQueryGroup(subqueries=[s[i] for s in subqueries]) for i in range(num_grps)]
        for i, g in enumerate(grps):
            for j, g_2 in enumerate(grps):
                if j > i and g_2.has(g.subqueries[0].parent_ptr):
                    g.parent_ptr = g_2
                    g_2.child_ptrs.append(g)
        # for i, g in enumerate(grps):
        #     print(f"Group {i}: {id(g)}")
        #     for subq in g.subqueries:
        #         print(f"{subq}")
        #     print(f" Parent: {id(g.parent_ptr) if g.parent_ptr else None}, Children: {[id(c) for c in g.child_ptrs]}")
        return grps[-1]
