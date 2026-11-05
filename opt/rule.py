from dataclasses import dataclass, field
from typing import Any, Dict, Set, Tuple, Optional
from enum import Enum
import logging


logger = logging.getLogger(__name__)

class Scope(Enum):
    ROW = 1   # Maps 1 row -> 1 row (Proj, Filter, SemProj, SemGroup)
    SET = 2   # Maps N rows -> 1 row or Reorders (Aggr, Sort, SemAggr, SemSort)
    CROSS = 3 # (Join, SemJoin)
    SCAN = 4

@dataclass
class OperatorProps:
    """Universal properties for any logical operator."""
    scope: Scope
    op: str
    requires_cols: Set[str] = field(default_factory=set)    # the minimal set of columns required to execute this operator
    yields_cols: Set[str]   = field(default_factory=set)    # all columns that are guaranteed to be present after executing this operator (including unchanged columns)
    params: Dict[str, Any] = field(default_factory=dict)

    def calculate_out_schema(self, in_schema: Set[str]) -> Set[str]:
        """Calculates the S_out based on S_in."""
        if not self.requires_cols.issubset(in_schema):
            raise ValueError(f"Invalid Plan: Missing required columns. Needed {self.requires_cols}, got {in_schema}")

        return self.yields_cols


def check_row_sort(parent: OperatorProps, child: OperatorProps, initial_schema: Set[str]) -> bool:
    if parent.scope == Scope.ROW and child.op in {"sort", "sem_sort"}:
        # If first sort then perform ROW ops, check sort key column is still present after ROW ops.
        try:
            parent_out_schema = parent.calculate_out_schema(initial_schema)
            if child.requires_cols.issubset(parent_out_schema):
                return True
        except ValueError:
            return False
    elif child.scope == Scope.ROW and parent.op in {"sort", "sem_sort"}:
        # Must be true
        return True
    return False


def check_filter_aggr(parent: OperatorProps, child: OperatorProps, initial_schema: Set[str]) -> bool:
    if parent.op in {"filter"} and child.op in {"aggr", "sem_aggr"} and parent.requires_cols == child.requires_cols:
        return True
    elif child.op in {"filter"} and parent.op in {"aggr", "sem_aggr"} and child.requires_cols == parent.requires_cols:
        return True
    return False


def check_row_cross(
        parent: OperatorProps,
        child: OperatorProps,
        initial_schema: Set[str],
        initial_schema_right: Optional[Set[str]] = None
    ):
    logger.debug(f"Parent {parent} Child {child}")
    logger.debug(f"Initial schema for left table: {initial_schema}, right table: {initial_schema_right}")
    if not initial_schema_right:
        return False, None

    if parent.scope == Scope.ROW and child.scope == Scope.CROSS:
        # If first join then perform ROW ops
        # check if ROW ops can be applied to any of the two join tables, i.e., the required columns for the join condition are still present after ROW ops.
        try:
            parent_out_schema = parent.calculate_out_schema(initial_schema)
            if child.requires_cols.issubset(parent_out_schema):
                return True, 'left'
        except ValueError:
            try:
                parent_out_schema_right = parent.calculate_out_schema(initial_schema_right)
                if child.requires_cols.issubset(parent_out_schema_right):
                    return True, 'right'
            except ValueError:
                return False, None
    elif child.scope == Scope.ROW and parent.scope == Scope.CROSS:
        # If first perform ROW ops then join
        if child.op in {"sem_proj"} and child.params['new_col'] in parent.requires_cols:
            return False, None
        else:
            return True, None
    return False, None


def is_immutable(
        parent: OperatorProps,
        child: OperatorProps,
        initial_schema: Set[str],
        initial_schema_right: Optional[Set[str]] = None
    ):
    """
    Universally determines if Parent(Child(Data)) can become Child(Parent(Data)).
    """
    if parent.scope != Scope.ROW or child.scope != Scope.ROW:
        # a. ROW can swap with sort / sem-sort
        row_sort = check_row_sort(parent, child, initial_schema)
        row_cross = check_row_cross(parent, child, initial_schema, initial_schema_right)
        filter_aggr = check_filter_aggr(parent, child, initial_schema)
        if row_sort or filter_aggr:
            return True
        if row_cross[0]:
            if row_cross[1] is not None:
                return True, row_cross[1]
            else: return True
        else: return False
        
    try:
        parent_out_schema = parent.calculate_out_schema(initial_schema)
        # Then Child executes SECOND
        child_out_schema = child.calculate_out_schema(parent_out_schema)
    except ValueError as e:
        # If calculating the schema raises a ValueError, a required column was missing.
        logger.debug(f"Schema reachability check failed: {e}")
        return False

    return True