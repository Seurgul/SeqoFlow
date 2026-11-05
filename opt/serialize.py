import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from opt.logic_op import *
from config.config import HOME
from schema import get_table_schema

SCHEMA_VERSION = 1

CANONICAL_OPS = {
    "scan",
    "filter",
    "projection",
    "join",
    "aggr",
    "sort",
    "sem_filter",
    "sem_proj",
    "sem_group",
    "sem_aggr",
    "sem_sort",
    "sem_join",
}

INTERNAL_TO_CANONICAL_OP = {
    "proj": "projection",
}

CANONICAL_REQUIRED_PARAMS = {
    "scan": {"table_dir"},
    "filter": {"predicate"},
    "projection": {"yields"},
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


def _to_canonical_op(op_type: str) -> str:
    return INTERNAL_TO_CANONICAL_OP.get(op_type, op_type)


def _must_have_keys(node_dict: Dict[str, Any], keys: List[str]) -> None:
    missing = [k for k in keys if k not in node_dict]
    if missing:
        raise ValueError(f"Malformed plan node. Missing required keys {missing}: {node_dict}")


def _validate_node_dict(node_dict: Dict[str, Any]) -> None:
    _must_have_keys(node_dict, ["op_type", "requires", "yields", "params"])

    op_type = node_dict["op_type"]
    if op_type not in CANONICAL_OPS:
        raise ValueError(f"Unknown operator type '{op_type}'. Allowed: {sorted(CANONICAL_OPS)}")

    if not isinstance(node_dict["params"], dict):
        raise ValueError(f"Node params must be a dict for op '{op_type}'")

    required_params = CANONICAL_REQUIRED_PARAMS[op_type]
    missing_params = sorted(required_params - set(node_dict["params"].keys()))
    if missing_params:
        raise ValueError(f"Operator '{op_type}' missing required params {missing_params}")

    if op_type in {"join", "sem_join"}:
        _must_have_keys(node_dict, ["left", "right"])
        _validate_node_dict(node_dict["left"])
        _validate_node_dict(node_dict["right"])
    else:
        child = node_dict.get("child")
        if child is not None:
            _validate_node_dict(child)


def _to_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _node_to_dict(node: LogicalOperator) -> Dict:
    op_type = _to_canonical_op(node.op_props.op)
    node_dict = {
        "oid": node.oid,
        "op_type": op_type,
        "requires": sorted(list(node.op_props.requires_cols)),
        "yields": sorted(list(node.op_props.yields_cols)),
        "params": dict(node.op_props.params),
    }

    if node.op_props.scope == Scope.CROSS:
        node_dict['left'] = _node_to_dict(node.left_child)
        node_dict['right'] = _node_to_dict(node.right_child)
    elif node.child:
        node_dict['child'] = _node_to_dict(node.child)

    return node_dict


def dump_plan(root_node: Any, filepath: str) -> None:
    plan_dict = _node_to_dict(root_node)
    _validate_node_dict(plan_dict)
    tree_dict = {
        "schema_version": SCHEMA_VERSION,
        "plan": plan_dict,
    }

    with open(filepath, 'w') as f:
        json.dump(tree_dict, f, indent=2)


def _table_columns(table_dir: str) -> List[str]:
    """Column names from the CSV header, or from schema/<db>/<table>.json when the CSV is absent."""
    path = HOME / table_dir
    if path.exists():
        return pd.read_csv(path, nrows=0).columns.tolist()
    return get_table_schema(path.parent.name, path.stem)


def _dict_to_node(node_dict: Dict, parent_node: Optional[LogicalOperator] = None) -> Any:
    _validate_node_dict(node_dict)
    op_type = node_dict["op_type"]
    params = dict(node_dict["params"])
    requires = set(node_dict["requires"])
    yields = set(node_dict["yields"])
    oid = node_dict.get("oid", None)

    match op_type:
        case "scan":
            node = TableScan(
                parent=parent_node,
                table_name=params.get("table_name", ""),
                in_cols=_table_columns(params["table_dir"]),
                op_props=OperatorProps(
                    op="scan",
                    scope=Scope.SCAN,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "projection":
            node = Projection(
                parent=parent_node,
                child=None,
                out_cols=_to_list(params.get("yields", list(yields))),
                op_props=OperatorProps(
                    op="proj",
                    scope=Scope.ROW,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_proj":
            node = SemanticProjection(
                parent=parent_node,
                child=None,
                out_cols=_to_list(params.get("yields", list(yields))),
                op_props=OperatorProps(
                    op="sem_proj",
                    scope=Scope.ROW,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "filter":
            node = Filter(
                parent=parent_node,
                child=None,
                filter_cols=_to_list(params.get("filter_cols", list(requires))),
                op_props=OperatorProps(
                    op="filter",
                    scope=Scope.ROW,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_filter":
            node = SemanticFilter(
                parent=parent_node,
                child=None,
                filter_cols=_to_list(params.get("filter_cols", list(requires))),
                op_props=OperatorProps(
                    op="sem_filter",
                    scope=Scope.ROW,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "aggr":
            node = Aggregation(
                parent=parent_node,
                child=None,
                group_by_cols=_to_list(params.get("group_by")),
                out_col=_to_list(params.get("out_col", list(yields))),
                op_props=OperatorProps(
                    op="aggr",
                    scope=Scope.SET,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_aggr":
            node = SemanticAggregation(
                parent=parent_node,
                child=None,
                group_by_cols=_to_list(params.get("group_by")),
                out_col=str(params.get("out_col", next(iter(yields)) if yields else "")),
                op_props=OperatorProps(
                    op="sem_aggr",
                    scope=Scope.SET,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sort":
            node = Sort(
                parent=parent_node,
                child=None,
                order_by_col=str(params.get("sort_key", params.get("order_by"))),
                op_props=OperatorProps(
                    op="sort",
                    scope=Scope.SET,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_sort":
            node = SemanticSort(
                parent=parent_node,
                child=None,
                order_by_col=str(params["order_by"]),
                op_props=OperatorProps(
                    op="sem_sort",
                    scope=Scope.SET,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "join":
            node = Join(
                parent=parent_node,
                left_child=None,
                right_child=None,
                op_props=OperatorProps(
                    op="join",
                    scope=Scope.CROSS,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_join":
            node = SemanticJoin(
                parent=parent_node,
                left_child=None,
                right_child=None,
                op_props=OperatorProps(
                    op="sem_join",
                    scope=Scope.CROSS,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case "sem_group":
            node = SemanticGroupBy(
                parent=parent_node,
                child=None,
                in_col=str(params["group_by"]),
                out_col=str(params["out_col"]),
                op_props=OperatorProps(
                    op="sem_group",
                    scope=Scope.SET,
                    requires_cols=requires,
                    yields_cols=yields,
                    params=params,
                ),
                oid=oid,
            )
        case _:
            raise ValueError(f"Unknown operator type: {op_type}")

    # Recursively build and link children
    if node.op_props.scope == Scope.CROSS:
        node.left_child = _dict_to_node(node_dict['left'], parent_node=node)
        node.right_child = _dict_to_node(node_dict['right'], parent_node=node)
    else:
        child_dict = node_dict.get('child')
        if child_dict is not None:
            node.child = _dict_to_node(child_dict, parent_node=node)

    return node


def load_plan(filepath: str) -> Any:

    with open(filepath, 'r') as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("Plan file must contain a JSON object payload")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {payload.get('schema_version')}. Expected {SCHEMA_VERSION}")

    plan_dict = payload.get("plan")
    if plan_dict is None:
        raise ValueError("Plan payload must contain a top-level 'plan' field")

    return _dict_to_node(plan_dict, parent_node=None)