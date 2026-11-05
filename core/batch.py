import pandas as pd
from typing import Protocol, List, Dict, Union


class Batch:
    def __init__(self, idx: List[int], data: pd.DataFrame, is_eof: bool = False, tables: List[str] = None):
        self.idx = idx
        self.data = data
        self.is_eof = is_eof
        self.bitmap = [[1 for _ in range(len(data))]]
        self.tables = tables
        self.meta = [(i,) for i in range(len(data))]
        self.fingerprint = []

    def _sync_bitmap(self):
        """Update the respective bitmap based on the meta"""
        # print("Before sync", self.bitmap)
        if self.meta == []:
            for which in range(len(self.bitmap)):
                self.bitmap[which] = [0 for _ in range(len(self.bitmap[which]))]
            return
        l = len(self.meta[0])
        for which in range(l):
            unique_vals = set()
            for m in self.meta:
                unique_vals.add(m[which])
            new_bm = [1 if u in unique_vals else 0 for u in range(len(self.bitmap[which]))]
            self.bitmap[which] = new_bm
        # print("After sync", self.bitmap)


    def _update_bm(self, old_bm, mask, which):
        assert sum(old_bm[which]) == len(mask), f"Bitmap length not matched, {old_bm[which]}, {mask}"
        new_bm = []
        flip_ = []
        i = 0
        for j in range(len(old_bm[which])):
            if old_bm[which][j]:
                new_bm.append(int(mask[i]))
                if mask[i] == 0:
                    flip_.append(j)
                i += 1
            else:
                new_bm.append(0)
        self.bitmap[which] = new_bm
        return flip_

def _merge(table1: List[str], bm1,
           table2: List[str], bm2):
    """
    bm1, bm2: bitmap after Join pruning.
    """
    tables = table1 + table2
    bm = bm1 + bm2
    return tables, bm
