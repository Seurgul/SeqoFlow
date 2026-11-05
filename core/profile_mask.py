"""Sample-local predicate bitmaps and tuple lineage used only for costing.

Each bitmap position has explicit source identities, so moving a predicate across
a join never relies on the order (or multiplicity) of the joined rows.
"""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class ProfileMask:
    tables: tuple
    rows: tuple
    bitmap: tuple

    def __post_init__(self):
        if len(self.rows) != len(self.bitmap):
            raise ValueError("Profile bitmap and row domain have different lengths")
        if len(set(self.tables)) != len(self.tables):
            raise ValueError("Profile relations require distinct table identities")

    @classmethod
    def from_batch(cls, batch):
        tables = tuple(batch.tables or ())
        # A materialized Source has one new row identity even when it contains columns from several original tables. Those columns share that row.
        materialized = len(batch.idx) == 1 and len(tables) > 1
        rows = tuple(
            tuple((batch.idx[0 if materialized else i],
                   meta[0 if materialized else i]) for i in range(len(tables)))
            for meta in batch.meta
        )
        return cls(tables, rows, (True,) * len(rows))

    @property
    def count(self):
        return sum(self.bitmap)

    def surviving_rows(self):
        return tuple(row for row, keep in zip(self.rows, self.bitmap) if keep)

    def project(self, tables):
        """Keep predicate dependencies, preserving identities and decisions."""
        indices = [self.tables.index(table) for table in tables]
        decisions = {}
        for row, keep in zip(self.rows, self.bitmap):
            key = tuple(row[i] for i in indices)
            if key in decisions and decisions[key] != keep:
                raise ValueError("Predicate decisions disagree for the same source rows")
            decisions[key] = keep
        return ProfileMask(tuple(tables), tuple(decisions), tuple(decisions.values()))

    def intersect(self, predicate):
        indices = [self.tables.index(table) for table in predicate.tables]
        decisions = dict(zip(predicate.rows, predicate.bitmap))
        result = []
        for row, keep in zip(self.rows, self.bitmap):
            if not keep:
                result.append(False)
                continue
            key = tuple(row[i] for i in indices)
            if key not in decisions:
                raise ValueError("Operator profile does not cover the candidate's source rows")
            result.append(bool(keep and decisions[key]))
        return ProfileMask(self.tables, self.rows, tuple(result))

    def cross(self, other):
        rows = tuple(left + right for left, right in
                     product(self.surviving_rows(), other.surviving_rows()))
        return ProfileMask(self.tables + other.tables, rows, (True,) * len(rows))

    def distinct_count(self, tables):
        indices = [self.tables.index(table) for table in tables]
        return len({tuple(row[i] for i in indices) for row in self.surviving_rows()})


def record_profile(op, mask, work=0):
    """Freeze sample statistics; normal execution must not replace this sample."""
    fields = ('sel', 'input_sz', 'output_sz', 'left_sel', 'right_sel',
              'input_left_sz', 'input_right_sz', 'prefill', 'decode', 'lat')
    op._profile_sample = {name: getattr(op, name) for name in fields if hasattr(op, name)}
    op._profile_sample.update(mask=mask, work=work)
