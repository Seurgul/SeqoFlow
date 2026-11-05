from dataclasses import dataclass, field
import itertools
from platform import node
from typing import List, Optional, Any, Dict
from copy import deepcopy
import logging

from opt.rule import OperatorProps, Scope
from opt.rule import is_immutable as _is_immutable
from opt.memo import get_memoizer


logger = logging.getLogger(__name__)

@dataclass(eq=False)
class LogicalOperator:
    def children(self) -> List['LogicalOperator']:
        return []

    def insert_child(self, children: List['LogicalOperator']):
        raise NotImplementedError(f"This operator does not support inserting children : {type(self)}")

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(eq=False)
class Placeholder(LogicalOperator):
    parent: Optional[LogicalOperator] = None
    child: Optional[LogicalOperator] = None
    in_cols: List[str] = field(default_factory=list)
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SCAN, op="placeholder", requires_cols=set(), yields_cols=set()))
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child] if self.child else []

    def insert_child(self, children: List['LogicalOperator']):
        self.child = children[0] if len(children) > 0 else None
        if self.child:
            self.child.parent = self


@dataclass(eq=False)
class FuseOp(LogicalOperator):
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    fused_op_alias: List[str] = field(default_factory=list)
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.ROW, op="fuse_filter", requires_cols=set(), yields_cols=set()))
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("FuseOp expects exactly one child.")
        self.child = children[0]
        self.child.parent = self

    def perm(self):
        """
        Enum all possible permutations in the fused filters
        """
        from itertools import permutations
        return list(permutations(self.fused_filters_alias))


@dataclass(eq=False)
class TableScan(LogicalOperator):
    table_name: str
    in_cols: List[str]
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SCAN, op="scan", requires_cols=set(), yields_cols=set()))
    child = None
    parent: Optional[LogicalOperator] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return []

    def insert_child(self, children: List['LogicalOperator']):
        return


@dataclass(eq=False)
class Projection(LogicalOperator):
    out_cols: List[str]
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.ROW, op="proj", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("Projection expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class Filter(LogicalOperator):
    filter_cols: List[str]
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.ROW, op="filter", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("Filter expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class Join(LogicalOperator):
    left_child: LogicalOperator = None
    right_child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None

    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.CROSS, op="join", requires_cols=set(), yields_cols=set()))
    oid: Optional[int] = None
    alias: Optional[str] = None

    def children(self) -> List['LogicalOperator']:
        return [self.left_child, self.right_child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 2 or len(children) == 1:
            raise ValueError("Join expects exactly two children.")
        self.left_child = children[0]
        self.right_child = children[1]
        self.left_child.parent = self
        self.right_child.parent = self


@dataclass(eq=False)
class Aggregation(LogicalOperator):
    group_by_cols: List[str]
    out_col: List[str]

    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SET, op="aggr", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("Aggregation expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class Sort(LogicalOperator):
    order_by_col: str
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SET, op="sort", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("Sort expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticFilter(LogicalOperator):
    filter_cols: List[str]
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.ROW, op="sem_filter", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("SemanticFilter expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticProjection(LogicalOperator):
    out_cols: List[str]
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.ROW, op="sem_proj", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("SemanticProjection expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticGroupBy(LogicalOperator):
    in_col: str
    out_col: str
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SET, op="sem_group", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("SemanticGroupBy expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticAggregation(LogicalOperator):
    group_by_cols: List[str] = field(default_factory=list)
    out_col: str = ""

    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SET, op="sem_aggr", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("SemanticAggregation expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticSort(LogicalOperator):
    order_by_col: str
    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.SET, op="sem_sort", requires_cols=set(), yields_cols=set()))
    child: LogicalOperator = None
    parent: Optional[LogicalOperator] = None
    alias: Optional[str] = None
    oid: Optional[int] = None

    def children(self) -> List['LogicalOperator']:
        return [self.child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 1:
            raise ValueError("SemanticSort expects exactly one child.")
        self.child = children[0]
        self.child.parent = self


@dataclass(eq=False)
class SemanticJoin(LogicalOperator):
    left_child: LogicalOperator = None
    right_child: LogicalOperator = None

    op_props: OperatorProps = field(default_factory=lambda: OperatorProps(scope=Scope.CROSS, op="sem_join", requires_cols=set(), yields_cols=set()))
    parent: Optional[LogicalOperator] = None
    oid: Optional[int] = None
    alias: Optional[str] = None

    def children(self) -> List['LogicalOperator']:
        return [self.left_child, self.right_child]

    def insert_child(self, children: List['LogicalOperator']):
        if len(children) > 2 or len(children) == 1:
            raise ValueError("SemanticJoin expects exactly two children.")
        self.left_child = children[0]
        self.right_child = children[1]
        self.left_child.parent = self
        self.right_child.parent = self


class SrcPartition:
    def __init__(self, root: LogicalOperator, src_node: LogicalOperator, parent: Optional[LogicalOperator] = None, pbr: Optional[str] = None, ch_ptrs: List[Optional[LogicalOperator]] = [None, None]):
        self.root = root
        self.src_node = src_node
        self.parent_ptr: Optional[LogicalOperator] = parent
        self.pbr: Optional[str] = pbr
        self.ch_ptrs: List[Optional[LogicalOperator]] = ch_ptrs

    def snapshot(self):
        node_map = {}
        def dfs_clone(old_node):
            new_node = deepcopy(old_node)
            node_map[id(old_node)] = new_node
            new_ch = []
            for child in old_node.children():
                new_child = dfs_clone(child)
                new_ch.append(new_child)
            new_node.insert_child(new_ch)
            return new_node
        new_root = dfs_clone(self.root)
        return SrcPartition(new_root, node_map[id(self.src_node)], self.parent_ptr, self.pbr, self.ch_ptrs)

    def print_plan(self):
        output = []
        def _print(node, indent=0):
            if isinstance(node, Placeholder):
                output.append(" " * indent + f"Placeholder ({id(node)})")
            else:
                output.append(" " * indent + f"{node.op_props.op}" + f" (id: {node.op_props.params['op_name']})")
            for child in node.children():
                _print(child, indent + 2)
        _print(self.root)
        return f"parent ptr : {id(self.parent_ptr) if self.parent_ptr else None}\n" + f"child ptrs: {[id(ch) for ch in self.ch_ptrs]}\n" + "\n".join(output)

    def walk_post_order(self):
        def _walk(node: LogicalOperator):
            for child in node.children():
                yield from _walk(child)
            yield node
        return _walk(self.root)

    def populate_props(self):
        for node in self.walk_post_order():
            if node.op_props.op == 'scan':
                node.op_props.requires_cols = set()
                node.op_props.yields_cols = set(node.in_cols)

            elif node.op_props.op in {'proj'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = set(node.op_props.params['columns']).intersection(child_props.yields_cols)
                node.op_props.yields_cols = set(node.op_props.params['columns']).union(child_props.yields_cols)

            elif node.op_props.op in {'sem_proj'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["key"]}.intersection(child_props.yields_cols)
                node.op_props.yields_cols = {node.op_props.params['new_col']}.union(child_props.yields_cols)

            elif node.op_props.op in {'filter', 'sem_filter'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["key"]}
                node.op_props.yields_cols = child_props.yields_cols

            elif node.op_props.op in {'join', 'sem_join'}:
                left_props = node.left_child.op_props
                right_props = node.right_child.op_props
                node.op_props.requires_cols = {node.op_props.params["left_on"], node.op_props.params["right_on"]}
                node.op_props.yields_cols = left_props.yields_cols.union(right_props.yields_cols)

            elif node.op_props.op in {'aggr', 'sem_aggr'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params.get("group_by", None), node.op_props.params["aggr_col"]}
                node.op_props.yields_cols = child_props.yields_cols.union({node.op_props.params["out_col"]})

            elif node.op_props.op in {'sort', 'sem_sort'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["order_by"]}
                node.op_props.yields_cols = child_props.yields_cols

            elif node.op_props.op == 'sem_group':
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["group_by"]}
                node.op_props.yields_cols = {node.op_props.params["out_col"]}.union(child_props.yields_cols)

            elif node.op_props.op == 'placeholder':
                # Same as scan
                node.op_props.requires_cols = set()
                node.op_props.yields_cols = set(node.in_cols)


class Plan:
    def __init__(self, root: LogicalOperator):
        self.root = root
        self.alias_to_op: Dict[str, LogicalOperator] = {}


    def walk(self):
        def _walk(node: LogicalOperator):
            yield node
            for child in node.children():
                yield from _walk(child)
        return _walk(self.root)


    def walk_post_order(self):
        def _walk(node: LogicalOperator):
            for child in node.children():
                yield from _walk(child)
            yield node
        return _walk(self.root)


    def print_plan(self):
        output = []
        def _print(node, indent=0):
            if isinstance(node, Placeholder):
                output.append(" " * indent + f"Placeholder")
            else:
                output.append(" " * indent + f"{node.op_props.op}" + f" (id: {id(node)})")
            for child in node.children():
                _print(child, indent + 2)
        _print(self.root)
        return "\n".join(output)

    def assign_oid(self):
        for node in self.walk():
            node.oid = id(node)

    def populate_props(self):
        for node in self.walk_post_order():
            if node.op_props.op == 'scan':
                node.op_props.requires_cols = set()
                node.op_props.yields_cols = set(node.in_cols)

            elif node.op_props.op in {'proj'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = set(node.op_props.params["columns"]).intersection(child_props.yields_cols)
                node.op_props.yields_cols = set(node.op_props.params["columns"]).union(child_props.yields_cols)

            elif node.op_props.op in {'sem_proj'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["key"]}.intersection(child_props.yields_cols)
                node.op_props.yields_cols = {node.op_props.params["new_col"]}.union(child_props.yields_cols)

            elif node.op_props.op in {'filter', 'sem_filter'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["key"]}
                node.op_props.yields_cols = child_props.yields_cols

            elif node.op_props.op in {'join', 'sem_join'}:
                left_props = node.left_child.op_props
                right_props = node.right_child.op_props
                node.op_props.requires_cols = {node.op_props.params["left_on"], node.op_props.params["right_on"]}
                node.op_props.yields_cols = left_props.yields_cols.union(right_props.yields_cols)

            elif node.op_props.op in {'aggr', 'sem_aggr'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params.get("group_by", None), node.op_props.params["aggr_col"]}
                node.op_props.yields_cols = child_props.yields_cols.union({node.op_props.params["out_col"]})

            elif node.op_props.op in {'sort', 'sem_sort'}:
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["order_by"]}
                node.op_props.yields_cols = child_props.yields_cols

            elif node.op_props.op == 'sem_group':
                child_props = node.child.op_props
                node.op_props.requires_cols = {node.op_props.params["group_by"]}
                node.op_props.yields_cols = {node.op_props.params["out_col"]}.union(child_props.yields_cols)

            elif node.op_props.op == 'placeholder':
                # Same as scan
                node.op_props.requires_cols = set()
                node.op_props.yields_cols = set(node.in_cols)


    def is_immutable(self, parent: LogicalOperator, child: LogicalOperator):
        if child.op_props.scope == Scope.CROSS :
            return _is_immutable(parent.op_props, child.op_props, child.left_child.op_props.yields_cols, child.right_child.op_props.yields_cols)
        elif parent.op_props.scope == Scope.CROSS:
            return _is_immutable(parent.op_props, child.op_props, parent.left_child.op_props.yields_cols, parent.right_child.op_props.yields_cols)
        else:
            if child.op_props.op == 'scan':
                return False
            else:
                return _is_immutable(parent.op_props, child.op_props, child.children()[0].op_props.yields_cols)

    def _copy_node(self, node: LogicalOperator) -> LogicalOperator:
        # Deep copy a node with null parent and child ptr.
        node_copy = deepcopy(node)
        node_copy.parent = None
        node_copy.children_list = []
        node_copy.oid = node.oid
        return node_copy

    def _copy_partition(self, sink_node: LogicalOperator):
        """Create a deep copy of the sink partition rooted at sink_node. The copied partition is disconnected from the original plan and returned as a new Plan, the parent ptr and two child ptrs.
        """

        ch_ptrs = [None, None]
        sink_copy = self._copy_node(sink_node)
        layer = [(sink_node, sink_copy)]
        pos = 0
        while layer != []:
            new_layer = []
            for node, node_copy in layer:
                new_ch = []
                for child in node.children():
                    if isinstance(child, Placeholder) or isinstance(child, TableScan) or child.op_props.scope == Scope.CROSS:
                        ch_ptrs[pos] = child
                        new_ch.append(Placeholder(in_cols=child.op_props.yields_cols))
                    else:
                        # ch_ptrs[pos] = None
                        child_copy = self._copy_node(child)
                        new_ch.append(child_copy)
                        new_layer.append((child, child_copy))

                    if ch_ptrs[1-pos] is None:
                        pos = 1 - pos

                node_copy.insert_child(new_ch)
            layer = new_layer

        parent = sink_node.parent
        pbr = None
        if parent and parent.op_props.op in {"join", "sem_join"}:
            pbr = "left" if parent and parent.left_child == sink_node else "right"
        root_plchd = Placeholder()
        root_plchd.insert_child([sink_copy])
        plan_copy = Plan(root_plchd)
        logger.debug(f"copy partition : {plan_copy.print_plan()}")
        logger.debug(f"ch_ptrs: left {ch_ptrs[0].op_props.op}, right {ch_ptrs[1].op_props.op}")
        return plan_copy, parent, pbr, ch_ptrs


    def _pushup_sink_partition(self, sink_node: LogicalOperator):
        """
        Push up a sink partition until no more immutability can be applied.
        """
        if sink_node.op_props.scope != Scope.CROSS:
            raise ValueError("Sink tree can only be extracted from a CROSS operator.")

        new_par, parent, _pbr, ch_ptrs = self._copy_partition(sink_node)

        def _pushup_node(parent: LogicalOperator, child: LogicalOperator, pbr=None, cbr=None):
            pp = parent.parent
            assert pp is not None, "Parent of the partition root cannot be null."
            logger.debug(f"PP: {pp.op_props.op if pp else None}, pbr: {pbr}, cbr: {cbr}")
            if pbr == 'left':
                """
                      pp
                    /    \
                    parent sibling
                    |
                    child
                    |
                    ...
                """
                sibling = pp.right_child
                parent.insert_child(child.children())
                child.insert_child([parent])
                pp.insert_child([child] + [sibling])
            elif pbr == 'right':
                """
                      pp
                    /    \
                    sibling parent
                            |
                            child
                            |
                            ...
                """
                sibling = pp.left_child
                parent.insert_child(child.children())
                child.insert_child([parent])
                pp.insert_child([sibling] + [child])
            else:
                if parent.op_props.scope == Scope.CROSS:
                    if cbr == 'left':
                        """
                            pp
                            |
                            parent
                            /   \
                            child  sibling
                        """
                        sibling = parent.right_child
                        parent.insert_child(child.children() + [sibling])
                        child.insert_child([parent])
                        pp.insert_child([child])
                    elif cbr == 'right':
                        """
                            pp
                            |
                            parent
                            /   \
                        sibling child
                        """
                        sibling = parent.left_child
                        parent.insert_child([sibling] + child.children())
                        child.insert_child([parent])
                        pp.insert_child([child])
                else:
                    """
                        pp
                        |
                        parent
                        |
                        child
                    """
                    parent.insert_child(child.children())
                    child.insert_child([parent])
                    pp.insert_child([child])

            return child

        def _try_pushup():
            pu = False
            disable_ = {"proj"}
            sink = new_par.root.children()[0]

            # A node is active only when :
            # 1. It is not SCAN, Projection or Filter
            # 2. It becomes a fresh child of the sink
            # 3. Its last pushup is successful and below the sink
            active = {node: False for node in new_par.walk()}
            fail_history = {node: False for node in new_par.walk()}

            # The next set of active ops
            layer = []
            for ch in sink.children():
                if ch.op_props.scope == Scope.SCAN:
                    continue
                if ch.op_props.op in disable_:
                    cur_node = ch
                    while True:
                        # Pass down to find the first active child
                        if cur_node.op_props.scope == Scope.SCAN:
                            break
                        if cur_node.op_props.op not in disable_:
                            active[cur_node] = True
                            layer.append(cur_node)
                            break
                        cur_node = cur_node.children()[0]
                else:
                    active[ch] = True
                    layer.append(ch)
            logger.debug(f"Next layer for pushup: {[node.op_props.op for node in layer]}, {len(layer)}")

            while len(layer) > 0:
                for ch in layer:
                    new_par.populate_props()
                    node = ch.parent
                    im = self.is_immutable(node, ch)

                    logger.debug(f" immutability for pushup: Parent({node.op_props.op}) and Child({ch.op_props.op}): {im}")

                    if not isinstance(im, bool):
                        im = im[0]
                    if im:
                        pu = True
                        cbr, pbr = None, None
                        if node.op_props.scope == Scope.CROSS:
                            cbr = 'left' if node.left_child == ch else 'right'
                        if node.parent and node.parent.op_props.scope == Scope.CROSS:
                            pbr = 'left' if node.parent.left_child == node else 'right'

                        _pushup_node(node, ch, pbr=pbr, cbr=cbr)

                        # If last pushup is successful:
                        # 1. If the parent node before is a sink, update its new child as active
                        # 2. If the parent node before is below the sink, update the pushed up node as active
                        # 3. If any of the above is proj / filter, trace down to find the first active child and update it as active
                        if node == sink:
                            fresh_child = node.children()[0] if cbr=='left' else node.children()[1]
                            active[ch] = False
                            if fresh_child.op_props.scope != Scope.SCAN:
                                if fresh_child.op_props.op in disable_:
                                    cur_node = fresh_child
                                    while True:
                                        if cur_node.op_props.scope == Scope.SCAN:
                                            break
                                        if cur_node.op_props.op not in disable_ and fail_history[cur_node] == False:
                                            active[cur_node] = True
                                            break
                                        cur_node = cur_node.children()[0]
                                elif fail_history[fresh_child] == False:
                                    active[fresh_child] = True
                            else: active[ch] = False
                        else:
                            active[ch] = True
                    else:
                        # If last pushup is not successful, update the child of the pushed-up node as active
                        fail_history[ch] = True
                        cc = ch.children()[0]
                        active[ch] = False
                        if cc.op_props.scope != Scope.SCAN:
                            if cc.op_props.op in disable_:
                                cur_node = cc
                                while True:
                                    # Pass down to find the first active child
                                    if cur_node.op_props.scope == Scope.SCAN:
                                        break
                                    if cur_node.op_props.op not in disable_ and fail_history[cur_node] == False:
                                        active[cur_node] = True
                                        break
                                    cur_node = cur_node.children()[0]
                            elif fail_history[cc] == False:
                                active[cc] = True
                        else: active[ch] = False

                    logger.debug(
                            "After pushup attempt:\n"
                            f"{new_par.print_plan()}"
                        )
                    logger.debug(f"Fail history: {[node.op_props.op + ':' + str(fail) for node, fail in fail_history.items()]}")

                # determine the next layer as all active nodes
                layer = [node for node, is_active in active.items() if is_active]
                logger.debug(f"Next layer for pushup: {[node.op_props.op for node in layer]}, {len(layer)}")
                assert len(layer) <= 2, f"Each layer must have at most 2 ops, got {len(layer)}"

            return pu

        _try_pushup()
        logger.debug(f"Final pushed-up partition:\n{new_par.print_plan()}")

        return new_par, parent, _pbr, ch_ptrs


    def _merge_partition(self, new_par, parent: LogicalOperator, pbr, ch_ptrs):
        """
        Merge the new parition into the original plan by replacing parts of the original plan.
        """
        actual_root = new_par.root.children()[0]
        if parent is None:
            self.root = actual_root
            pass
        elif pbr is None:
            parent.insert_child([actual_root])
        elif pbr == 'left':
            parent.insert_child([actual_root] + [parent.right_child])
        elif pbr == 'right':
            parent.insert_child([parent.left_child] + [actual_root])

        # Find a Join
        def probe_join(node):
            cur_node = node
            while cur_node.op_props.scope != Scope.CROSS:
                if len(cur_node.children()) == 0:
                    return None    # reach a leaf node but no Join found
                cur_node = cur_node.children()[0]
            return cur_node
        join_node = probe_join(new_par.root)
        assert join_node is not None, "The new partition must contain a Join operator."
        left_branch, right_branch = join_node.left_child, join_node.right_child

        cur_node = left_branch
        while True:
            if isinstance(cur_node, Placeholder):
                cur_node = cur_node.parent
                break
            if cur_node.children() == []:
                break
            cur_node = cur_node.children()[0]

        if cur_node.op_props.scope == Scope.CROSS:
            right_sibling = cur_node.children()[1]
            cur_node.insert_child([ch_ptrs[0], right_sibling])
        else:
            cur_node.insert_child([ch_ptrs[0]])

        cur_node = right_branch
        while True:
            if isinstance(cur_node, Placeholder):
                cur_node = cur_node.parent
                break
            if cur_node.children() == []:
                break
            cur_node = cur_node.children()[0]

        if cur_node.op_props.scope == Scope.CROSS:
            left_sibling = cur_node.children()[0]
            cur_node.insert_child([left_sibling, ch_ptrs[1]])
        else:
            cur_node.insert_child([ch_ptrs[1]])

        return


    def pushup(self, root_node: LogicalOperator):
        """
        Bottom-up push up all partitions
        """

        # Find a Join
        def probe_join(node):
            cur_node = node
            while cur_node.op_props.scope != Scope.CROSS:
                if len(cur_node.children()) == 0:
                    return None    # reach a leaf node but no Join found
                cur_node = cur_node.children()[0]
            return cur_node

        cur_node = probe_join(root_node)
        if cur_node is None: return  # nothing to do

        # Find the left and right child partitions
        assert len(cur_node.children()) == 2, "A Join must have exactly two children."
        left_child, right_child = cur_node.children()[0], cur_node.children()[1]

        left_sink = probe_join(left_child)
        right_sink = probe_join(right_child)

        if left_sink is not None:
            self.pushup(left_sink)
        if right_sink is not None:
            self.pushup(right_sink)

        self.populate_props()

        new_par, parent, _pbr, ch_ptrs = self._pushup_sink_partition(cur_node)
        # Merge reformed partition into this plan
        self._merge_partition(new_par, parent, _pbr, ch_ptrs)
        self.populate_props()

        # Assign unique oid, this should NOT be modified anymore afterwards
        for node in self.walk():
            node.oid = id(node)
            # TODO

        logger.debug(f"After pushup at Join (id: {id(cur_node)}):\n{self.print_plan()}")


    # ========================= Pushdown  =========================

    def _swap_in_place(self, parent: LogicalOperator, child: LogicalOperator, pp: LogicalOperator):
        cc = child.children()
        parent.insert_child(cc)
        child.insert_child([parent])
        pp.insert_child([child])

    def _copy_src_partition(self, src_node: LogicalOperator):
        ch_ptrs = src_node.children()
        src_copy = self._copy_node(src_node)
        cur_node = src_node.parent
        ch_copy = src_copy
        pbr = None
        if cur_node and cur_node.op_props.op in {"join", "sem_join"}:
            pbr = 'left' if cur_node.left_child == src_node else 'right'
        else:
            while cur_node:
                new_node = self._copy_node(cur_node)
                new_node.insert_child([ch_copy])
                ch_copy = new_node
                tmp = cur_node
                cur_node = cur_node.parent
                if cur_node and cur_node.op_props.scope == Scope.CROSS:
                    pbr = 'left' if cur_node.left_child == tmp else 'right'
                    break
                if cur_node is None:
                    break
                if isinstance(cur_node, Placeholder):
                    cur_node = None
                    break

        root_plchd = Placeholder()
        root_plchd.insert_child([ch_copy])
        self.populate_props()

        if src_copy.op_props.op == "scan":
            pass
        else:
            src_copy.insert_child(
                [Placeholder(in_cols=ch_ptrs[0].op_props.yields_cols),
                Placeholder(in_cols=ch_ptrs[1].op_props.yields_cols)])
            src_copy.children()[0].op_props.yields_cols = ch_ptrs[0].op_props.yields_cols
            src_copy.children()[1].op_props.yields_cols = ch_ptrs[1].op_props.yields_cols
        parent = cur_node
        par = SrcPartition(root=root_plchd, src_node=src_copy, parent=parent, pbr=pbr, ch_ptrs=ch_ptrs)
        return par


    def _copy_full_plan(self):
        """
        node_map : old node id -> new node
        """
        node_map = {}

        def dfs_clone(old_node):
            new_node = deepcopy(old_node)
            node_map[id(old_node)] = new_node
            new_ch = []
            for child in old_node.children():
                new_child = dfs_clone(child)
                new_ch.append(new_child)
            new_node.insert_child(new_ch)
            return new_node

        new_root = dfs_clone(self.root)
        logger.debug(f"Full plan copied:\n{[(nid, id(node)) for nid, node in node_map.items()]}")
        return new_root, node_map


    def _merge_src_partition(self, new_par, src_copy, parent: LogicalOperator, pbr, ch_ptrs):
        actual_root = new_par.root.children()[0]
        if pbr is None:
            if parent is not None:
                parent.insert_child([actual_root])
            else:
                self.root = actual_root
        elif pbr == 'left':
            parent.insert_child([actual_root] + [parent.right_child])
        elif pbr == 'right':
            parent.insert_child([parent.left_child] + [actual_root])

        if src_copy.op_props.op == "scan":
            return
        left_branch, right_branch = src_copy.children()[0], src_copy.children()[1]
        if left_branch.op_props.scope == Scope.SCAN and right_branch.op_props.scope == Scope.SCAN:
            src_copy.insert_child(ch_ptrs)
            return

        if left_branch.op_props.scope != Scope.SCAN:
            cur_node = left_branch
            while not isinstance(cur_node, Placeholder):
                cur_node = cur_node.children()[0]
            cur_node.parent.insert_child([ch_ptrs[0]])
            if right_branch.op_props.scope == Scope.SCAN:
                src_copy.insert_child([left_branch, ch_ptrs[1]])
                return
        if right_branch.op_props.scope != Scope.SCAN:
            cur_node = right_branch
            while not isinstance(cur_node, Placeholder):
                cur_node = cur_node.children()[0]
            cur_node.parent.insert_child([ch_ptrs[1]])
            if left_branch.op_props.scope == Scope.SCAN:
                src_copy.insert_child([ch_ptrs[0], right_branch])
        return


    def _pushdown_in_partition(self, partition: SrcPartition):
        src_node = partition.src_node

        def pushdown_filter(type):
            # get all filter ops from the source partition in bottom-up order
            partition.populate_props()
            filters = []
            cur_node = src_node
            while not isinstance(cur_node, Placeholder):
                if cur_node.op_props.op in {type}:
                    filters.append(cur_node)
                cur_node = cur_node.parent

            # try to push down each of them
            for f in filters:
                ch = f.children()[0]
                while ch.op_props.scope not in {Scope.CROSS, Scope.SCAN} and self.is_immutable(f, ch):
                    logger.debug(f"{f.op_props.op} and {ch.op_props.op} are immutable for pushdown.")
                    partition.populate_props()
                    # if type == "sem_filter" and ch.op_props.op == "filter":
                    #     break   # semantic filter should not be pushed down below a normal filter
                    pp = f.parent
                    self._swap_in_place(f, ch, pp)
                    ch = f.children()[0]

                logger.debug(f"{f.oid} cannot be pushed down further beyond {ch.oid}.")

        def pushdown_sem_proj():
            # get all sem_proj ops from the source partition in bottom-up order
            partition.populate_props
            sem_projs = []
            cur_node = src_node
            while not isinstance(cur_node, Placeholder):
                if cur_node.op_props.op == "sem_proj":
                    sem_projs.append(cur_node)
                cur_node = cur_node.parent

            # try to push down each of them
            for sp in sem_projs:
                partition.populate_props()
                ch = sp.children()[0]
                while ch.op_props.scope not in {Scope.CROSS, Scope.SCAN} and self.is_immutable(sp, ch):
                    if ch.op_props.op in {"filter", "sem_filter"}:
                        break
                    pp = sp.parent
                    self._swap_in_place(sp, ch, pp)
                    ch = sp.children()[0]

        # 1. push down a single node in bottom-up order
        pushdown_sem_proj()
        # logger.debug(f"After pushing down semantic projects:\n{partition.print_plan()}")
        pushdown_filter("sem_filter")
        # logger.debug(f"After pushing down semantic filters:\n{partition.print_plan()}")
        pushdown_filter("filter")
        logger.info(f"After pushing down normal filters:\n{partition.print_plan()}")

        # 2. Get consecutive dependency segments
        def get_alias(node: LogicalOperator):
            """
            Assign a unique alias to this node, replacing aliases inherited by copies.
            """
            node.alias = f"op_{id(node)}"
            self.alias_to_op[node.alias] = node
            # print(node.alias)
            return node.alias

        def get_dep_seg():
            partition.populate_props()
            cur_node = src_node.parent
            anchor = cur_node
            cs_sf = []
            all_fused = []
            parent = None
            while True:
                if isinstance(cur_node, Placeholder) or cur_node.op_props.scope in {Scope.CROSS, Scope.SET}:
                    if len(cs_sf) >= 1:
                        fuse = FuseOp(fused_op_alias=cs_sf.copy())
                        logger.debug(f"Fusing ops {[self.alias_to_op[a].op_props.op for a in cs_sf]} into {fuse.op_props.op} with alias {fuse.fused_op_alias}.")
                        fuse.insert_child([all_fused[-1] if len(all_fused) > 0 else src_node])
                        all_fused.append(fuse)
                    break

                alias = get_alias(cur_node)
                if id(cur_node) == id(anchor):
                    cs_sf.append(alias)
                elif not (self.is_immutable(cur_node, anchor)):
                    logger.debug(f"Node {cur_node.op_props.op} is in the same segment as anchor {anchor.op_props.op}.")
                    cs_sf.append(alias)
                else:
                    if len(cs_sf) >= 1:
                        fuse = FuseOp(fused_op_alias=cs_sf.copy())
                        logger.debug(f"Fusing ops {[self.alias_to_op[a].op_props.op for a in cs_sf]} into {fuse.op_props.op} with alias {fuse.fused_op_alias}.")
                        fuse.insert_child([all_fused[-1] if len(all_fused) > 0 else src_node])
                        cs_sf = [alias]
                        all_fused.append(fuse)
                        anchor = cur_node
                    else:
                        cs_sf = [alias]

                logger.debug(f"{cur_node.alias}, {cs_sf}")
                cur_node = cur_node.parent

            parent = cur_node
            return all_fused, parent

        dep_seg, par_ptr = get_dep_seg()
        logger.info(f"Dependency segment for pushdown: {[[self.alias_to_op[alias].op_props.params['op_name'] for alias in a.fused_op_alias] for a in dep_seg]}.")

        # 3. Apply permutation across all fused segments and return all resulting partition copies.
        partition_seeds = []

        def apply_prune_heuristic(dep_seg, parent_ptr):
            ch_ = src_node
            # 1. If the segment only has 1 op and it is a filter, place it right above the src
            for fuse_op in dep_seg[:]:
                if len(fuse_op.fused_op_alias) == 1:
                    op = self.alias_to_op[fuse_op.fused_op_alias[0]]
                    if op.op_props.op in {"filter"}:

                        op_parent = op.parent
                        op_children = op.children()
                        op_parent.insert_child(op_children)
                        for child in op_children:
                            child.parent = op_parent

                        ch_parent = ch_.parent
                        ch_parent.insert_child([op])
                        op.parent = ch_parent
                        op.insert_child([ch_])
                        ch_.parent = op
                        ch_ = op
                        dep_seg.remove(fuse_op)

            # 2. If the segement only has 1 op and it is a sem_proj, place it right below the parent
            par_tmp = parent_ptr
            for fuse_op in dep_seg[:]:
                if len(fuse_op.fused_op_alias) == 1:
                    op = self.alias_to_op[fuse_op.fused_op_alias[0]]
                    if op.op_props.op in {"sem_proj"}:
                        op_children = op.children()
                        op_parent = op.parent
                        op_parent.insert_child(op_children)
                        for child in op_children:
                            child.parent = op_parent

                        children = par_tmp.children() if par_tmp else []
                        par_tmp.insert_child([op])
                        op.parent = par_tmp
                        op.insert_child(children)
                        for child in children:
                            child.parent = op
                        par_tmp = op
                        dep_seg.remove(fuse_op)

            return par_tmp, ch_

        p_, ch_ = apply_prune_heuristic(dep_seg, par_ptr)
        logger.debug(f"dep_seg: {[f.op_props.op + ':' + str(f.fused_op_alias) for f in dep_seg]}, p: {p_.op_props.op if p_ else None}, ch: {ch_.op_props.op if ch_ else None}")
        logger.debug(f"After prune, {partition.print_plan()}")

        if len(dep_seg) <= 1:
            partition_seeds.append(partition.snapshot())
            return partition_seeds

        for ordered_fuse_op in itertools.permutations(dep_seg):
            logger.debug(f"Applying permutation {[str(op.fused_op_alias) for op in ordered_fuse_op]}")
            first_ = self.alias_to_op[ordered_fuse_op[0].fused_op_alias[-1]]
            p_.insert_child([first_])
            first_.parent = p_
            for j, fuse_op in enumerate(ordered_fuse_op):
                for i in range(len(fuse_op.fused_op_alias)-1, 0, -1):
                    f = self.alias_to_op[fuse_op.fused_op_alias[i]]
                    f.insert_child([self.alias_to_op[fuse_op.fused_op_alias[i-1]]])
                    self.alias_to_op[fuse_op.fused_op_alias[i-1]].parent = f
                if j != len(ordered_fuse_op)-1:
                    first_ = self.alias_to_op[ordered_fuse_op[j+1].fused_op_alias[-1]]
                    self.alias_to_op[fuse_op.fused_op_alias[0]].insert_child([first_])
                    first_.parent = self.alias_to_op[fuse_op.fused_op_alias[0]]
                else:
                    self.alias_to_op[fuse_op.fused_op_alias[0]].insert_child([ch_])
                    ch_.parent = self.alias_to_op[fuse_op.fused_op_alias[0]]

            par_copy = partition.snapshot()
            partition_seeds.append(par_copy)

        self.alias_to_op.clear()   # clear the alias mapping for the next pushdown step
        return partition_seeds


    def _pushdown_btw_partition(self, partition: SrcPartition, src_node: LogicalOperator=None):
        """
        Iteratively push down ops in bottom-up order through the src Join
        - par : the partition copy from _copy_src_partition()
        - return : all partition results after pushdown
        """
        src_node = partition.src_node if src_node is None else src_node
        partition_seeds = []

        cur_node = src_node.parent
        while not isinstance(cur_node, Placeholder):
            partition.populate_props()
            im = self.is_immutable(cur_node, src_node)
            if not isinstance(im, bool):
                im, cbr = im
                if im:
                    logger.debug(f"im : {im}, cbr: {cbr}")
                    if cbr == 'left':
                        left_ch, right_ch = src_node.children()[0], src_node.children()[1]
                        pp = cur_node.parent
                        cur_node.insert_child([left_ch])
                        src_node.insert_child([cur_node, right_ch])
                        pp.insert_child([src_node])
                    elif cbr == 'right':
                        left_ch, right_ch = src_node.children()[0], src_node.children()[1]
                        pp = cur_node.parent
                        cur_node.insert_child([right_ch])
                        src_node.insert_child([left_ch, cur_node])
                        pp.insert_child([src_node])
                    else:
                        raise ValueError("Join should have a clear child branch for pushdown.")

                    # If the pushdown is successful, create a snapshot of the current partition state
                    par_copy = partition.snapshot()
                    logger.debug(f"Pushdown btw partition applied {par_copy.print_plan()}")
                    partition_seeds.append(par_copy)
                    success = True
                    cur_node = src_node.parent
            else:
                logger.debug(f"im : {im}")
                par_copy = partition.snapshot()
                partition_seeds.append(par_copy)
                break

        return partition_seeds


    def _get_kth_layer_join(self, k):
        """
        Returns a list of all Joins (nodes with 2 children) at the k-th layer
        in top-down order. k >= 1.
        """
        frontier = [self.root]

        for current_k in range(1, k + 1):
            next_frontier = []
            current_joins = []

            for start_node in frontier:
                curr = start_node
                while curr is not None and len(curr.children()) == 1:
                    curr = curr.children()[0]
                if curr is not None and len(curr.children()) == 2:
                    current_joins.append(curr)
                    next_frontier.extend(curr.children())
                if curr.op_props.scope == Scope.SCAN:
                    current_joins.append(curr)

            if current_k == k:
                return current_joins

            frontier = next_frontier
            if not frontier:
                return []
        return []


    # Position vectors (paper, Appendix A.2) used to refuse duplicate merges.

    @staticmethod
    def _label(node):
        return node.op_props.params["op_name"]

    @staticmethod
    def _is_anchor(node):
        return node.op_props.scope in (Scope.CROSS, Scope.SCAN)

    @staticmethod
    def _up_chain(anchor):
        """Labels of the operators above `anchor`, nearest first, up to the next join or the root."""
        chain, cur = [], anchor.parent
        while cur is not None and not isinstance(cur, Placeholder) and cur.op_props.scope != Scope.CROSS:
            chain.append(Plan._label(cur))
            cur = cur.parent
        return tuple(chain)

    def _anchor_state(self):
        """Fixed join/scan skeleton (anchor -> child anchors) and the chain above every anchor."""
        def below(node):
            while not self._is_anchor(node):
                node = node.children()[0]
            return node
        anchors = [n for n in self.walk() if self._is_anchor(n)]
        skeleton = {self._label(a): tuple(self._label(below(c)) for c in a.children()) for a in anchors}
        chains = {self._label(a): self._up_chain(a) for a in anchors}
        return skeleton, chains

    @staticmethod
    def _position_key(skeleton, chains):
        anchors = [a for a, ch in skeleton.items() if ch] or list(skeleton)
        key = []
        for j in sorted(anchors):
            arms = [chains[j]] + [tuple(reversed(chains[c])) for c in skeleton[j]]
            vec = sorted((op, 3 * r + b + 1) for b, arm in enumerate(arms) for r, op in enumerate(arm))
            key.append((j, tuple(vec)))
        return tuple(key)

    def _pushdown(self, input_plan, k):
        """
        Push down all possible ops in top-down order. This procedure starts from the plan state after pushup() and submits all possible pushdown transformed plans.
        """
        memo = get_memoizer()

        # 1. Find the next Join and create copy of its src partition
        all_src_nodes = input_plan._get_kth_layer_join(k)

        logger.debug(f"Start input plan: \n {input_plan.print_plan()}")
        logger.debug(f"Found {len(all_src_nodes)} joins at layer {k} for pushdown.")
        if len(all_src_nodes) == 0:
            if memo.admit(self._position_key(*input_plan._anchor_state())):
                memo.add(input_plan)
            return

        layer = [[input_plan._copy_src_partition(src)] for src in all_src_nodes]   # a layer is a list of partition variant sets

        # 2. Push down inside the partition. This should generate a set of partition copies as seeds.
        new_seeds = [[] for _ in range(len(layer))]
        for i, par_br in enumerate(layer):
            logger.debug(f"Pushdown branch {i}")
            for partition_seed in par_br:
                new_pars = input_plan._pushdown_in_partition(partition_seed)
                new_seeds[i].extend(new_pars)
                logger.debug(f"New seeds {len(new_pars)} in branch {i}")

        logger.debug(f"{len(new_seeds)} branches, {[len(seeds) for seeds in new_seeds]}")

        skeleton, chains = input_plan._anchor_state()
        new_plan_info = []
        for seed_comb in itertools.product(*new_seeds):
            # Refuse the merge (and memoization) if this plan's position vectors were seen before.
            comb_chains = dict(chains)
            comb_chains.update({self._label(s.src_node): self._up_chain(s.src_node) for s in seed_comb})
            if not memo.admit(self._position_key(skeleton, comb_chains)):
                continue
            new_root, node_map = input_plan._copy_full_plan()
            plan_copy = Plan(new_root)
            plan_copy.populate_props()
            src_node_ = []
            for seed in seed_comb:
                seed_copy = seed.snapshot()
                # logger.debug(f"Merge pushdown seed : \n {seed_copy.print_plan()}")
                # logger.debug(f"Parent ptr {(seed_copy.parent_ptr.op_props.op, id(seed_copy.parent_ptr)) if seed_copy.parent_ptr else None}, pbr: {seed_copy.pbr}, ch ptrs: {[(ch.op_props.op, id(ch)) for ch in seed_copy.ch_ptrs]}")
                plan_copy._merge_src_partition(seed_copy, seed_copy.src_node, node_map.get(id(seed_copy.parent_ptr), None), seed_copy.pbr, [node_map[id(ch)] for ch in seed_copy.ch_ptrs])
                src_node_.append(seed_copy.src_node)
            plan_copy.populate_props()
            memo.add(plan_copy)
            new_plan_info.append((plan_copy, src_node_))    # record the src nodes of the seeds that lead to this new plan for the next pushdown step

        # 3. Push down through Join,
        new_plans_next = []
        for new_plan, seed_src_nodes in new_plan_info:
            new_root, node_map = new_plan._copy_full_plan()
            plan_copy = Plan(new_root)
            plan_copy.populate_props()
            src_node_copies = [node_map[id(src_node)] for src_node in seed_src_nodes]
            new_seeds_2 = [None for _ in range(len(src_node_copies))]
            for j, src_node_copy in enumerate(src_node_copies):
                par = plan_copy._copy_src_partition(src_node_copy)
                logger.debug(f"Discovered seed \n {par.print_plan()}")
                new_pars = plan_copy._pushdown_btw_partition(par)

                # Store new seeds in the branch, still need to compute cartesian product and merge.
                new_seeds_2[j] = new_pars
            logger.debug(f"New seeds in {len(new_seeds_2)} branches after pushdown btw partition: {[len(seeds) for seeds in new_seeds_2]}")

            for seed_comb in itertools.product(*new_seeds_2):
                new_root, node_map_2 = plan_copy._copy_full_plan()
                new_plan_copy = Plan(new_root)
                new_plan_copy.populate_props()
                for seed in seed_comb:
                    seed_copy = seed.snapshot()
                    new_plan_copy._merge_src_partition(seed_copy, seed_copy.src_node, node_map_2.get(id(seed_copy.parent_ptr), None), seed_copy.pbr, [node_map_2[id(ch)] for ch in seed_copy.ch_ptrs])
                new_plan_copy.populate_props()
                # memo.add(new_plan_copy)
                new_plans_next.append(new_plan_copy)

        # 4. continue to next layer
        if len(new_plans_next) > 0:
            for new_plan in new_plans_next:
                self._pushdown(new_plan, k + 1)
        else:
            return


    def pushdown(self):
        self._pushdown(self, 1)
