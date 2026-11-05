import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum, auto
import logging
import functools
from copy import deepcopy
import numpy as np

from core.batch import Batch, _merge
from core.execution_mode import full_batch_execution
from core.profile_mask import ProfileMask, record_profile
from core.runtime_dump import observe_evaluations, observe_mask, observe_tokens, trace_operator, tracing_enabled
from core.base import Operator, TYPE_ID
from core.prompt_templates import *
from schema import get_table_schema
from config.randomize import get_batch_size
from config.config import __config__, __config_scan__, __config_join__, configured_source_rows

logger = logging.getLogger(__name__)

def __filter_broadcast__(func):
    @functools.wraps(func)
    def wrapper(self, payload, is_profile=False):
        if payload.is_eof:
            return func(self, payload, is_profile)
        if self.oid in payload.fingerprint:
            return payload
        if is_profile or full_batch_execution():
            return func(self, payload, is_profile)
        before_count = len(payload.data)
        if not hasattr(payload, '_raw_unfiltered'):
            payload._raw_unfiltered = deepcopy(payload)
        # Broadcast - batch_id is a list of joined batch ids. First , use _which_table to find which slot is the current operator processing. Then check if the batch_id has been cached in bms.
        # IF cached, prune

        which_ = self._which_table(payload.tables)
        batch_id = payload.idx[which_]
        if batch_id in self.bms:
            flip_ = self.bms[batch_id]
            cache_ = self.cache_.get(batch_id, set())
            print(f"Batch {batch_id} has cache for keys {cache_}")
            # Payload and metadata should be updated in-place
            drop_ = []
            for i, exist in enumerate(payload.bitmap[which_]):
                if exist and i in flip_:
                    payload.bitmap[which_][i] = 0
                    drop_.append(i)

            drop_set = set(drop_)
            drop_meta = [
                i for i, meta in enumerate(payload.meta)
                if meta[which_] in drop_set
            ]
            # meta and row should be in exactly the same order
            payload.meta = [meta for i, meta in enumerate(payload.meta) if i not in drop_meta]
            payload.data = payload.data.drop(payload.data.index[drop_meta])

        # Proceed to the operator to process the pruned batch, potentially update the cache and return the final result.
        result = func(self, payload, is_profile)
        if len(result.data) < before_count:
            result._filtered_op_ids = set(getattr(result, '_filtered_op_ids', ())) | {self.oid}
        return result
    return wrapper


def _replay_batch(batch, stream):
    """Recover rows pruned by filters that moved above the join."""
    needed = {op.oid for op in stream}
    if (hasattr(batch, '_raw_unfiltered') and
            any(oid not in needed for oid in getattr(batch, '_filtered_op_ids', ()))):
        return deepcopy(batch._raw_unfiltered), True
    return deepcopy(batch), False


def __proj_broadcast__(func):
    @functools.wraps(func)
    def wrapper(self, payload, is_profile=False):
        logger.debug(f"{len(payload.data)} rows tot")
        if payload.is_eof:
            return func(self, payload, is_profile)
        if self.oid in payload.fingerprint:
            return payload
        if is_profile:
            return func(self, payload, is_profile)

        which_ = self._which_table(payload.tables)
        batch_id = payload.idx[which_]
        new_payload = None
        if batch_id in self.cache_:
            cache_df = self.cache_[batch_id]
            # Only pass null rows to the op
            null_mask = cache_df["new_value"].isna().to_numpy()
            logger.debug(f"null mask {null_mask}")
            if all(not null_mask[meta[which_]] for meta in payload.meta):
                payload.data[self.new_col] = [cache_df.iloc[meta[which_]]['new_value']
                                              for meta in payload.meta]
                payload.fingerprint.append(self.oid)
                return payload
            new_payload = deepcopy(payload)
            no_cache, pos_mask = [], []
            cache_ = []
            for i, meta in enumerate(new_payload.meta):
                if null_mask[meta[which_]]:
                    no_cache.append(payload.data.iloc[i])
                    pos_mask.append(True)
                else:
                    cache_.append(cache_df.iloc[meta[which_]].to_dict())
                    pos_mask.append(False)
            logger.debug(f"{len(no_cache)} rows without cache, {len(cache_)} rows with cache")
            pos_mask = np.array(pos_mask, dtype=bool)

            new_payload.data = pd.DataFrame(no_cache)
            new_payload.meta = [meta for i, meta in enumerate(new_payload.meta) if pos_mask[i]]
            logger.debug(f"new payload {len(new_payload.data)}")
            ret = func(self, new_payload, is_profile)
            # patch back
            payload.data[self.new_col] = None
            logger.debug(f"{len(payload.data)} rows out")

            k, j = 0, 0
            m = 0
            for i in payload.data.index:
                if pos_mask[m]:
                    payload.data.at[i, self.new_col] = ret.data.to_dict(orient='records')[k][self.new_col]
                    k += 1
                else:
                    payload.data.at[i, self.new_col] = cache_[j]['new_value']
                    j += 1
                m += 1

            payload.fingerprint.append(self.oid)
            return payload
        else:
            return func(self, payload, is_profile)
    return wrapper


def _run_llm(llm, prompts, **kwargs):
    observe_evaluations(len(prompts))
    return llm.as_llm_call(prompts, **kwargs)


def _get_llm_resp(llm_result: Dict[str, Any]) -> List[str]:
    responses = llm_result.get("responses", []) if isinstance(llm_result, dict) else llm_result
    prefill = llm_result.get("input_tokens", 0) if isinstance(llm_result, dict) else 0
    decode = llm_result.get("output_tokens", 0) if isinstance(llm_result, dict) else 0
    lat = llm_result.get("runtime_sec", 0) if isinstance(llm_result, dict) else 0
    observe_tokens(prefill, decode, lat)
    return responses, prefill, decode, lat


class Sink(Operator):
    """
    Sink placeholder if the root is not blocking.
    """
    def __init__(self):
        self._oid = -1
        self.is_eof = False
        self.buffer: List[Batch] = []
        self.__type_id__ = TYPE_ID.SINK
        self.op_name = "Sink"

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False) -> Batch:
        if is_profile:
            if payload.is_eof:
                return Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)
            else:
                return payload

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            if len(self.buffer) == 0:
                all_data = pd.DataFrame()
            else:
                all_data = pd.concat([batch.data for batch in self.buffer])
            output = Batch(idx=[-1], data=all_data, is_eof=True)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output
        else:
            return None

class Source(Operator):
    """
    Placeholder to hold the yield of the children subqueries
    """
    def __init__(self, data, tables):
        self._oid = -1
        self.__type_id__ = TYPE_ID.SOURCE
        self.buffer: pd.DataFrame = data
        self.cursor = 0
        self.is_eof = False
        self.tables = tables
        self.table_name = '+'.join(tables) if isinstance(tables, list) else tables
        self.batch_size = get_batch_size()
        self.op_name = "Source"
        self.tot = len(self.buffer)

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch=None, is_profile=False) -> Batch:
        if self.cursor >= len(self.buffer):
            ret = Batch(idx=[-1], data=pd.DataFrame(), is_eof=True, tables=self.tables)
            if is_profile:
                self._profile_sample = {"output_sz": 0, "mask": ProfileMask.from_batch(ret)}
            return ret
        batch = self.buffer.iloc[self.cursor:min(self.cursor + self.batch_size, len(self.buffer))]
        if is_profile:
            batch = batch.copy(deep=True)
        ret =  Batch(idx=[self.cursor // self.batch_size], data=batch, tables=self.tables)
        if is_profile:
            self._profile_sample = {"output_sz": len(ret.data),
                                    "mask": ProfileMask.from_batch(ret)}
        if is_profile is False:
            self.cursor += self.batch_size
        return ret

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "output_sz": self.batch_size
        }

    def get_progress(self):
        return min(self.cursor, len(self.buffer)) / len(self.buffer) if len(self.buffer) > 0 else None

class Scan(Operator):
    @__config_scan__()
    def __init__(
            self,
            oid: int,
            table_dir: Optional[str] = None,
            table_data: Optional[pd.DataFrame]=None,
            table_name: Optional[str] = None,
            batch_size: Optional[int] = None,
            **kwargs
        ):
        self._oid = oid
        self.table = pd.read_csv(table_dir) if table_dir else table_data
        self.batch_size = batch_size or get_batch_size()
        self.cursor = 0
        self.is_eof = False
        self.__type_id__ = TYPE_ID.SCAN
        self.table_name = table_name
        self.tot = len(self.table)
        self.op_name = kwargs.get('op_name', None)

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch = None, is_profile=False) -> Batch:
        if self.cursor >= len(self.table):
            ret = Batch(idx=[-1], data=pd.DataFrame(), is_eof=True, tables=[self.table_name])
            if is_profile:
                self._profile_sample = {"output_sz": 0, "mask": ProfileMask.from_batch(ret)}
            return ret
        logger.info(f"Scan {self.op_name} processing {self.cursor} : {min(self.cursor + self.batch_size, len(self.table))} / {len(self.table)}")
        batch = self.table.iloc[self.cursor:min(self.cursor + self.batch_size, len(self.table))]
        if is_profile:
            batch = batch.copy(deep=True)
        ret =  Batch(idx=[self.cursor // self.batch_size], data=batch, tables=[self.table_name])
        if is_profile:
            self._profile_sample = {"output_sz": len(ret.data),
                                    "mask": ProfileMask.from_batch(ret)}
        if is_profile is False:
            self.cursor += self.batch_size
        return ret

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "output_sz": self.batch_size
        }

    def get_progress(self):
        return min(self.cursor, self.tot) / self.tot if self.tot > 0 else None

class Filter(Operator):
    def __init__(self, oid: int, predicate, key, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.key = key
        self.predicate = predicate
        self.op_name = kwargs.get('op_name', None)
        self.sel = 0.0  # for profiling, record the selectivity of the filter
        self.input_sz = 0
        self.output_sz = 0
        self.__type_id__ = TYPE_ID.FILTER

        self.bms = {}   # batch_id -> [flipped positions]
        self.cache_ = {}  # batch_id -> cached positions
        self.tables = []

    @property
    def oid(self) -> int:
        return self._oid

    def _update_bms(self, batch_id: int, flip_):
        if batch_id not in self.bms:
            self.bms[batch_id] = set(flip_)
        else:
            self.bms[batch_id].update(flip_)

    def _which_table(self, tables: List[str]):
        idx = 0
        for i, t in enumerate(tables):
            if self.tables[0] == t:
                idx = i
                break
        return idx

    @trace_operator
    @__config__()
    @__filter_broadcast__
    def process(self, payload: Batch, is_profile=False) -> Batch:
        logger.info(f"Filter {self.op_name} processing")
        preserve_input = is_profile or full_batch_execution()
        if preserve_input:
            original_ = deepcopy(payload)
        if payload.is_eof:
            return payload
        batch_sz = len(payload.data)
        if batch_sz == 0:
            if is_profile:
                self.sel = self.input_sz = self.output_sz = 0
                record_profile(self, ProfileMask.from_batch(original_).project(self.tables))
            return payload
        # payload.data = payload.data.query(self.predicate)
        configured = getattr(payload, '_configured_filter_decisions', {}).get(self.oid)
        if configured is None:
            decision_mask = payload.data.eval(self.predicate)
        else:
            source, chosen = configured
            decision_mask = pd.Series(
                [meta[source] in chosen for meta in payload.meta],
                index=payload.data.index)
        payload.data = payload.data[decision_mask]
        which_ = self._which_table(payload.tables)

        unique_ = {}
        decision_mask = decision_mask.to_list()
        for i, meta in enumerate(payload.meta):
            key = meta[which_]
            if key not in unique_:
                unique_[key] = [i]
            else:   unique_[key].append(i)
        reduce_decision_mask = [False] * len(unique_)

        for i, key in enumerate(unique_.keys()):
            reduce_decision_mask[i] = any(decision_mask[idx] for idx in unique_[key])
        flip_ = payload._update_bm(payload.bitmap, reduce_decision_mask, which_)

        if not preserve_input:
            if payload.idx[which_] not in self.cache_:
                self.cache_[payload.idx[which_]] = set()
            self.cache_[payload.idx[which_]].union(set(unique_.keys()))
            self._update_bms(payload.idx[which_], flip_)

        new_meta = []
        for i, keep in enumerate(decision_mask):
            # Prune all meta with i not kept
            if keep:
                new_meta.append(payload.meta[i])
        payload.meta = new_meta
        payload._sync_bitmap()


        self.sel = len(payload.data) / batch_sz if batch_sz > 0 else 0
        self.input_sz = batch_sz
        self.output_sz = len(payload.data)

        if preserve_input:
            domain = ProfileMask.from_batch(original_)
            mask = ProfileMask(domain.tables, domain.rows, tuple(decision_mask)).project(self.tables)
            if is_profile:
                record_profile(self, mask, work=len(mask.rows))
            else:
                observe_mask(self, mask, input_mask=domain,
                             output_mask=domain.intersect(mask))
            original_.fingerprint.append(self.oid)
            return original_

        payload.fingerprint.append(self.oid)
        return payload

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "sel": self.sel,
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
        }

class Projection(Operator):
    def __init__(self, oid: int, columns: List[str], **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.columns = columns
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.PROJECTION

        self.input_sz = None
        self.output_sz = None

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False) -> Batch:
        if payload.is_eof:
            return payload
        payload.data = payload.data[self.columns]
        payload.fingerprint.append(self.oid)
        if is_profile:
            self.input_sz = len(payload.data)
            self.output_sz = len(payload.data)
        return payload

    def _get_profile(self):
        return {
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
        }

class Join(Operator):
    def __init__(self, oid: int, left_on, right_on, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.left_on = left_on
        self.right_on = right_on
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.JOIN

        self.tables = [None, None]

        self.left_buffer: Optional[Batch] = None
        self.right_buffer: List[Batch] = []
        self.right_cursor = 0
        self.right_eof = False

        self.backup = {}

        # Profile
        self.left_sel = 0
        self.right_sel = 0
        self.input_left_sz = 0
        self.input_right_sz = 0
        self.output_sz = 0

    @property
    def oid(self) -> int:
        return self._oid

    def _backup(self):
        self.backup = {
            'left_buffer': self.left_buffer,
            'right_buffer_len': len(self.right_buffer),
            'right_cursor': self.right_cursor,
            'right_eof': self.right_eof
        }

    def _restore(self):
        self.left_buffer = self.backup.get('left_buffer', None)
        right_buffer_len = self.backup.get('right_buffer_len', 0)
        self.right_buffer = self.right_buffer[:right_buffer_len]
        self.right_cursor = self.backup.get('right_cursor', 0)
        self.right_eof = self.backup.get('right_eof', False)

    def _which_table(self, tab, tables: List[str]):
        idx = None
        for i, t in enumerate(tables):
            if tab == t:
                idx = i
                break
        assert idx is not None, f"Join operator cannot find the table for table {tab} in tables {tables}"
        return idx

    @trace_operator
    @__config_join__()
    def process(
            self,
            payload: Batch,
            side: str,
            is_profile=False,
            upstream_ckpt=None,
            **kwargs
        ) -> Optional[Batch]:
        """Join uses stream handling. It should always keep 1 batch in the left buffer and stream through the right batches, and yield 1 joined batch at a time."""

        logger.info(f"Join processing")

        left_stream, right_stream = upstream_ckpt if upstream_ckpt is not None else ([], [])
        if side == 'left':
            if payload.is_eof:
                # No more left batch will arrive, i.e. join is completed.
                return Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)

            # Checkpoint stream op
            compl = True
            for op in left_stream[::-1]:
                if op.oid in payload.fingerprint:
                    continue
                else:
                    compl = False
                    compl_payload = op.process(payload, is_profile=is_profile)
            # Directly cover the left buffer
            if is_profile is False:
                self.left_buffer = payload if compl else compl_payload
            else:
                self.left_buffer = payload
            # Here the operator should wait for the next right batch to be submitted, so it should not emit any output yet.
            return None
        elif side == 'right':
            if self.right_eof is False:
                self.right_buffer.append(payload)
                self.right_cursor += 1

            if payload.is_eof:
                self.right_eof = True
                self.right_cursor = 0
                return None
            assert self.left_buffer is not None, "Right batch arrives before left batch for join operator"
            # Emit immediately
            if self.right_eof:
                data = self.right_buffer[self.right_cursor].data
                r_idx = self.right_buffer[self.right_cursor].idx
                right_copy, restored_right = _replay_batch(
                    self.right_buffer[self.right_cursor], right_stream)
                self.right_cursor += 1
            else:
                data = payload.data
                r_idx = payload.idx
                right_copy, restored_right = _replay_batch(payload, right_stream)

            # Manage bitmaps, we have to deepcopy two batches
            left_copy, restored_left = _replay_batch(self.left_buffer, left_stream)

            compl = True
            for op in left_stream[::-1]:
                if op.oid in left_copy.fingerprint:
                    continue
                else:
                    compl = False
                    left_copy = op.process(left_copy, is_profile=is_profile)
            if (compl is False or restored_left) and is_profile is False:
                self.left_buffer = left_copy

            missing = [op for op in right_stream[::-1] if op.oid not in right_copy.fingerprint]

            if (missing and self.right_eof and is_profile is False and
                    all(op.__type_id__ in {TYPE_ID.FILTER, TYPE_ID.SEM_FILTER}
                        for op in missing)):
                ret = self._join_pair(left_copy, right_copy, is_profile, kwargs)
                if len(ret.data) == 0:
                    return ret
            for op in missing:
                right_copy = op.process(right_copy, is_profile=is_profile)
            if (missing or restored_right) and is_profile is False:
                self.right_buffer[self.right_cursor-1] = right_copy

            return self._join_pair(left_copy, right_copy, is_profile, kwargs)

    def _join_pair(self, left_copy, right_copy, is_profile, kwargs):
        l_tmp, r_tmp = deepcopy(left_copy), deepcopy(right_copy)
        left_values = l_tmp.data[self.left_on].tolist() if self.left_on in l_tmp.data else []
        right_values = r_tmp.data[self.right_on].tolist() if self.right_on in r_tmp.data else []

        new_tables, new_bm = _merge(l_tmp.tables, l_tmp.bitmap, r_tmp.tables, r_tmp.bitmap)

        l_tmp.data['_l'] = np.arange(len(l_tmp.data))
        r_tmp.data['_r'] = np.arange(len(r_tmp.data))

        # NOTE: This is a leaky way to apply join config.
        if kwargs['use_config']:
            left_sel = kwargs['left_sel']
            right_sel = kwargs['right_sel']
            left_source = self._which_table(self.tables[0], l_tmp.tables)
            right_source = self._which_table(self.tables[1], r_tmp.tables)
            left_chosen = configured_source_rows(
                self.dataset, self.plan_idx, self.op_name,
                l_tmp.tables[left_source], l_tmp.idx[left_source],
                len(l_tmp.bitmap[left_source]), left_sel)
            right_chosen = configured_source_rows(
                self.dataset, self.plan_idx, self.op_name,
                r_tmp.tables[right_source], r_tmp.idx[right_source],
                len(r_tmp.bitmap[right_source]), right_sel)
            data = l_tmp.data
            for i in range(len(data)):
                if l_tmp.meta[i][left_source] in left_chosen:
                    data.at[data.index[i], self.left_on] = "[JOIN_KEEP]"
                elif data.at[data.index[i], self.left_on] == "[JOIN_KEEP]":
                    data.at[data.index[i], self.left_on] = "[JOIN_PRUNE]"
            l_tmp.data = data

            data = r_tmp.data
            for i in range(len(data)):
                if r_tmp.meta[i][right_source] in right_chosen:
                    data.at[data.index[i], self.right_on] = "[JOIN_KEEP]"
                elif data.at[data.index[i], self.right_on] == "[JOIN_KEEP]":
                    data.at[data.index[i], self.right_on] = "[JOIN_PRUNE]"
            r_tmp.data = data

        if len(l_tmp.data) == 0 or len(r_tmp.data) == 0:
            joined_data = pd.DataFrame()
            new_meta = []
        else:
            joined_data = l_tmp.data.merge(r_tmp.data, how='inner', left_on=self.left_on, right_on=self.right_on)
            if kwargs['use_config']:
                joined_data[self.left_on] = [left_values[int(i)] for i in joined_data['_l']]
                joined_data[self.right_on] = [right_values[int(i)] for i in joined_data['_r']]

            new_meta = []
            for _, row in joined_data.iterrows():
                _l, _r = row['_l'], row['_r']
                new_meta.append(l_tmp.meta[_l] + r_tmp.meta[_r])
            joined_data.drop(columns=['_l', '_r'], inplace=True)

        left_sz, right_sz = len(l_tmp.data), len(r_tmp.data)
        filtered_left_sz = len(joined_data[self.left_on].unique()) if len(joined_data) > 0 else 0
        filtered_right_sz = len(joined_data[self.right_on].unique()) if len(joined_data) > 0 else 0
        self.left_sel = filtered_left_sz / left_sz if left_sz > 0 else 0
        self.right_sel = filtered_right_sz / right_sz if right_sz > 0 else 0
        self.input_left_sz = left_sz
        self.input_right_sz = right_sz
        self.output_sz = len(joined_data)

        ret = Batch(idx=l_tmp.idx + r_tmp.idx, data=joined_data)
        ret.bitmap = new_bm
        ret.tables = new_tables
        ret.meta = new_meta
        ret.fingerprint = l_tmp.fingerprint + r_tmp.fingerprint + [self.oid]
        ret._sync_bitmap()
        if is_profile or tracing_enabled():
            domain = ProfileMask.from_batch(l_tmp).cross(ProfileMask.from_batch(r_tmp))
            matches = set(ProfileMask.from_batch(ret).rows)
            mask = ProfileMask(domain.tables, domain.rows,
                               tuple(row in matches for row in domain.rows))
            if is_profile:
                mask = mask.project(self.tables)
                record_profile(self, mask, work=left_sz * right_sz)
            observe_mask(self, mask, input_mask=domain, output_mask=ProfileMask.from_batch(ret))
        return ret

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "left_sel": self.left_sel,
            "right_sel": self.right_sel,
            "input_left_sz": self.input_left_sz,
            "input_right_sz": self.input_right_sz,
            "output_sz": self.output_sz
        }


class Aggregation(Operator):
    def __init__(self, oid: int, group_by: str, agg_func: Dict[str, str], out_col: str, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.group_by = group_by
        self.agg_func = agg_func
        self.out_col = out_col
        self.tables = []
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.AGGREGATION

        self.buffer: List[Batch] = []
        self.is_eof = False

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False):
        """In pull-based execution, this operator should not yield any output until EOF is received"""
        if is_profile:
            # for profiling, we do not block but stream the immediate result for the current batch.
            if payload.is_eof:
                return Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)
            else:
                payload.data = payload.data.groupby(self.group_by).agg(self.agg_func).reset_index()
                return payload

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            # concat all batches in the buffer, aggregate them and emit 1 new batch
            if len(self.buffer) == 0:
                all_data = pd.DataFrame()
            else:
                all_data = pd.concat([batch.data for batch in self.buffer])

            aggregated_data = all_data.groupby(self.group_by).agg(self.agg_func).reset_index()
            output = Batch(idx=[-1], data=aggregated_data, tables=self.tables)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output
        else: return None

    def _get_profile(self):
        return super()._get_profile()

class Sort(Operator):
    def __init__(self, oid: int, order_by: str, desc: bool = False, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.sort_key = order_by
        self.desc = desc
        self.__type_id__ = TYPE_ID.SORT
        self.op_name = kwargs.get('op_name', None)
        self.tables = []

        self.buffer: List[Batch] = []
        self.is_eof = False

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False):
        if is_profile:
            # for profiling, we do not block but stream the immediate result for the current batch.
            if payload.is_eof:
                return Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)
            else:
                if len(payload.data) == 0:
                    return payload
                payload.data = payload.data.sort_values(by=self.sort_key, ascending=not self.desc)
                return payload

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            # concat all batches in the buffer, sort them and emit as new batches.
            if self.buffer == []:
                sorted_data = pd.DataFrame()
            else:
                all_data = pd.concat([batch.data for batch in self.buffer])
                sorted_data = all_data.sort_values(by=self.sort_key, ascending=(not self.desc))
            output = Batch(idx=[-1], data=sorted_data, tables=self.tables)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output

    def _get_profile(self):
        return super()._get_profile()

class SemanticFilter(Operator):
    def __init__(self, oid: int, key: str, predicate: str, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.predicate = predicate
        self.key = key
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.SEM_FILTER

        self.bms = {}   # batch_id -> [flipped positions]
        self.cache_ = {}  # batch_id -> cached positions
        self.cache_bm = {}  # batch_id -> bitmap of cached positions
        self.tables = []
        self.active = False

        self.llm_client = llm_client
        self.llm_args = llm_args or {}

        self.sel = 0  # for profiling, record the selectivity of the filter
        self.input_sz = 0
        self.output_sz = 0
        self.prefill = 0
        self.decode = 0
        self.lat = 0

    @property
    def oid(self) -> int:
        return self._oid

    def _update_bms(self, batch_id: int, flip_):
        if batch_id not in self.bms:
            self.bms[batch_id] = set(flip_)
        else:
            self.bms[batch_id].update(flip_)

    def _which_table(self, tables: List[str]):
        idx = 0
        for i, t in enumerate(tables):
            if self.tables[0] == t:
                idx = i
                break
        return idx

    @trace_operator
    @__config__()
    @__filter_broadcast__
    def process(self, payload: Batch, is_profile=False) -> Batch:
        self.active = True
        preserve_input = is_profile or full_batch_execution()
        if preserve_input:
            original_ = deepcopy(payload)
        if payload.is_eof:
            return payload

        # Reduce redundant rows
        # deduplicate payload meta on which
        which_ = self._which_table(payload.tables)
        unique_ = {}
        for i, meta in enumerate(payload.meta):
            key = meta[which_]
            if key not in unique_:
                unique_[key] = [i]
            else:   unique_[key].append(i)
        reduced_ = []
        cached_, no_cache_ = {}, {}
        for key, idxs in unique_.items():
            if not is_profile and key in self.cache_.get(payload.idx[which_], set()):
                cached_[key] = idxs
            else:
                no_cache_[key] = idxs
                reduced_.append(payload.data.iloc[idxs[0]])
        reduced_data = pd.DataFrame(reduced_)

        prompts = sem_filter_template(field=self.key, predicate=self.predicate, df=reduced_data)
        # print(prompts)
        llm_result = _run_llm(self.llm_client, prompts, **self.llm_args)

        responses, prefill, decode, lat = _get_llm_resp(llm_result)
        print(f"SemanticFilter {self.op_name} - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")

        if len(responses) != len(reduced_data):
            raise ValueError(f"SemanticFilter {self.op_name} expected {len(reduced_data)} responses, got {len(responses)}")

        # Broadcast
        print(len(payload.data), len(responses), len(unique_), len(cached_), len(no_cache_))
        brdcst_responses = [None] * len(payload.data)
        for j, key, idxs in zip(range(len(responses)), no_cache_.keys(), no_cache_.values()):
            for idx in idxs:
                brdcst_responses[idx] = responses[j]

        for j, (key, idxs) in enumerate(cached_.items()):
            for idx in idxs:
                brdcst_responses[idx] = self.cache_bm[payload.idx[which_]][key]

        decision_mask = [False] * len(unique_.keys())
        for i, key in enumerate(unique_.keys()):
            if key in cached_:
                decision_mask[i] = self.cache_bm[payload.idx[which_]][key]
            else:
                decision_mask[i] = str(responses[list(no_cache_.keys()).index(key)]).strip().lower().startswith("true")

        configured = getattr(payload, '_configured_filter_decisions', {}).get(self.oid)
        if configured is not None:
            source, chosen = configured
            decision_mask = [key in chosen for key in unique_]

        # decision_mask = [str(resp).strip().lower().startswith("true") for resp in responses]
        b_decision_mask = ([meta[source] in chosen for meta in payload.meta]
                           if configured is not None else
                           [str(resp).strip().lower().startswith("true") for resp in brdcst_responses])
        batch_sz = len(payload.data)
        payload.data = payload.data.loc[b_decision_mask]
        which_ = self._which_table(payload.tables)

        flip_ = payload._update_bm(payload.bitmap, decision_mask, which_)

        if not is_profile:
            if payload.idx[which_] not in self.cache_:
                self.cache_[payload.idx[which_]] = set()
            self.cache_[payload.idx[which_]].update(list(unique_.keys()))

            if payload.idx[which_] not in self.cache_bm:
                self.cache_bm[payload.idx[which_]] = {}
            self.cache_bm[payload.idx[which_]].update({key: resp for key, resp in zip(no_cache_.keys(), decision_mask)})

        if not preserve_input:
            self._update_bms(payload.idx[which_], flip_)

        new_meta = []
        for i, keep in enumerate(b_decision_mask):
            # Prune all meta with i not kept
            if keep:
                new_meta.append(payload.meta[i])
        payload.meta = new_meta
        payload.fingerprint.append(self.oid)

        payload._sync_bitmap()
        logger.info(f"{self.oid} yield {payload.meta}, {len(payload.data)}")

        self.sel = len(payload.data) / batch_sz if batch_sz > 0 else 0
        self.input_sz = batch_sz
        self.output_sz = len(payload.data)
        self.prefill = prefill
        self.decode = decode
        self.lat = lat

        if preserve_input:
            domain = ProfileMask.from_batch(original_)
            mask = ProfileMask(domain.tables, domain.rows, tuple(b_decision_mask)).project(self.tables)
            if is_profile:
                record_profile(self, mask, work=len(unique_))
            else:
                observe_mask(self, mask, input_mask=domain,
                             output_mask=domain.intersect(mask))
            original_.fingerprint.append(self.oid)
            return original_

        return payload

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "sel": self.sel,
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }

class SemanticProjection(Operator):
    def __init__(self, oid: int, key: str, new_col: str, instr: str, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.key = key
        self.new_col = new_col
        self.instr = instr
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.SEM_PROJECTION

        self.llm_client = llm_client
        self.llm_args = llm_args or {}
        self.tables = []
        self.cache_ = {}    # batch_id -> DataFrame
        self.active = False

        self.input_sz = 0
        self.output_sz = 0
        self.prefill = 0
        self.decode = 0
        self.lat = 0

    @property
    def oid(self) -> int:
        return self._oid

    def _which_table(self, tables: List[str]):
        idx = 0
        for i, t in enumerate(tables):
            if self.tables[0] == t:
                idx = i
                break
        return idx

    def _update_cache(self, batch_id: int, df: pd.DataFrame):
        """
        df : {row_id, new_value}
        """
        if batch_id not in self.cache_:
            self.cache_[batch_id] = df
        else:
            for row_id, new_value in df.itertuples(index=False):
                cached = self.cache_[batch_id]
                matching = cached['row_id'] == row_id
                if matching.any():
                    if pd.notna(new_value):
                        cached.loc[matching & cached['new_value'].isna(), 'new_value'] = new_value
                else:
                    self.cache_[batch_id] = pd.concat([self.cache_[batch_id], pd.DataFrame({'row_id': row_id, 'new_value': new_value})], ignore_index=True)

    @trace_operator
    @__config__()
    @__proj_broadcast__
    def process(self, payload: Batch, is_profile=False) -> Batch:
        self.active = True
        if payload.is_eof:
            return payload

        which_ = self._which_table(payload.tables)
        unique_ = {}
        for i, meta in enumerate(payload.meta):
            key = meta[which_]
            if key not in unique_:
                unique_[key] = [i]
            else:   unique_[key].append(i)
        reduced_ = []
        for key, idxs in unique_.items():
            reduced_.append(payload.data.iloc[idxs[0]])
        reduced_data = pd.DataFrame(reduced_)

        prompts = sem_proj_template(field=self.key, instr=self.instr, df=reduced_data)
        llm_result = _run_llm(self.llm_client, prompts, **self.llm_args)
        responses, prefill, decode, lat = _get_llm_resp(llm_result)
        print(f"SemanticProjection LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")

        if len(responses) != len(reduced_data):
            raise ValueError(f"SemanticProjection expected {len(reduced_data)} responses, got {len(responses)}")

        # Broadcast
        brdcst_responses = [None] * len(payload.data)
        for j, key, idxs in zip(range(len(responses)), unique_.keys(), unique_.values()):
            for idx in idxs:
                brdcst_responses[idx] = responses[j]

        if not is_profile:
            which_ = self._which_table(payload.tables)
            cache_df = pd.DataFrame([{
                "row_id": i,
                "new_value": None
            } for i in range(len(payload.bitmap[which_]))])
            idx = 0
            for i, meta in enumerate(payload.meta):
                row_id = meta[which_]
                cache_df.at[row_id, 'new_value'] = brdcst_responses[idx]
                idx += 1

            cache_df = pd.DataFrame(cache_df)
            self._update_cache(payload.idx[which_], cache_df)
            logger.debug(f"SemanticProjection cache updated for batch_id {payload.idx[which_]},  {(cache_df)}, {payload.bitmap[which_]}")

        payload.data.loc[:, self.new_col] = [str(resp).strip() for resp in brdcst_responses]
        payload.fingerprint.append(self.oid)

        self.input_sz = len(payload.data)
        self.output_sz = len(payload.data)
        self.prefill = prefill
        self.decode = decode
        self.lat = lat
        if is_profile:
            record_profile(self, ProfileMask.from_batch(payload), work=len(unique_))
        # logger.info(f"Output: {self._oid}\n{payload.data[self.new_col]}")
        return payload

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }

class SemanticGroupBy(Operator):
    def __init__(self, oid: int, group_by: str, out_col: str, instr: str, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.group_by = group_by
        self.out_col = out_col
        self.instr = instr
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.SEM_GROUPBY
        self.tables = []

        self.llm_client = llm_client
        self.llm_args = llm_args or {}
        self.buffer: List[Batch] = []
        self.is_eof = False

        self.input_sz = None
        self.output_sz = None
        self.prefill = None
        self.decode = None
        self.lat = None

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False) -> Batch:
        """For simplicity, we first ask the LLM to generate the group labels first, then call a row-wise LLM to assign each row with a group label."""

        if is_profile:
            return payload
            # if payload.is_eof:
            #     return payload
            # self.prefill, self.decode, self.lat = 0, 0, 0
            # self.input_sz = len(payload.data)
            # if self.input_sz == 0:
            #     payload.data[self.out_col] = []
            #     output = Batch(idx=[-1], data=payload.data, is_eof=True)
            #     return output
            # # Step 1: generate group labels
            # prompts = sem_groupby_template_1(group_by=self.group_by, instr=self.instr, df=payload.data)
            # llm_result = self.llm_client.llm_call(prompts, **self.llm_args)
            # responses, prefill, decode, lat = _get_llm_resp(llm_result)
            # print(f"SemanticGroupBy (group label generation) LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            # if len(responses) != 1:
            #     raise ValueError(f"SemanticGroupBy expected 1 response for group label generation, got {len(responses)}")
            # self.prefill += prefill
            # self.decode += decode
            # self.lat += lat

            # grp_labels = responses[0]
            # prompts = sem_groupby_template_2(group_by=self.group_by, instr=self.instr, grp_labels=grp_labels, df=payload.data)
            # llm_result = self.llm_client.llm_call(prompts, **self.llm_args)
            # responses, prefill, decode, lat = _get_llm_resp(llm_result)
            # print(f"SemanticGroupBy (group label assignment) LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            # if len(responses) != len(payload.data):
            #     raise ValueError(f"SemanticGroupBy expected {len(payload.data)} responses for group label assignment, got {len(responses)}")
            # self.prefill += prefill
            # self.decode += decode
            # self.lat += lat

            # if len(payload.data) > 0:
            #     payload.data[self.out_col] = [str(resp).strip() for resp in responses]
            # output = Batch(idx=[-1], data=payload.data)
            # self.output_sz = len(payload.data)
            # return output

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            if len(self.buffer) == 0:
                all_data = pd.DataFrame()
            else:
                all_data = pd.concat([batch.data for batch in self.buffer])

            if len(all_data) == 0:
                all_data[self.out_col] = []
                output = Batch(idx=[-1], data=all_data, is_eof=True)
                if is_profile is False:
                    self.buffer.clear()
                    self.is_eof = False
                return output

            # Step 1: generate group labels
            prompts = sem_groupby_template_1(group_by=self.group_by, instr=self.instr, df=all_data)
            llm_result = _run_llm(self.llm_client, prompts, **self.llm_args)
            responses, prefill, decode, lat = _get_llm_resp(llm_result)
            print(f"SemanticGroupBy (group label generation) LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            if len(responses) != 1:
                raise ValueError(f"SemanticGroupBy expected 1 response for group label generation, got {len(responses)}")

            grp_labels = responses[0]
            prompts = sem_groupby_template_2(group_by=self.group_by, instr=self.instr, grp_labels=grp_labels, df=all_data)
            llm_result = _run_llm(self.llm_client, prompts, **self.llm_args)
            responses, prefill, decode, lat = _get_llm_resp(llm_result)
            print(f"SemanticGroupBy (group label assignment) LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            if len(responses) != len(all_data):
                raise ValueError(f"SemanticGroupBy expected {len(all_data)} responses for group label assignment, got {len(responses)}")
            if len(all_data) > 0:
                all_data[self.out_col] = [str(resp).strip() for resp in responses]
            output = Batch(idx=[-1], data=all_data, tables=self.tables)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output
        else:
            return None

    def _get_profile(self):
        return {
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }

class SemanticAggregation(Operator):
    def __init__(self, oid: int, group_by: Optional[str], aggr_col: str, out_col: str, instr: str, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.group_by = group_by
        self.aggr_col = aggr_col
        self.out_col = out_col
        self.instr = instr
        self.op_name = kwargs.get('op_name', None)
        self.__type_id__ = TYPE_ID.SEM_AGGREGATION
        self.tables = []

        self.llm_client = llm_client
        self.llm_args = llm_args or {}
        self.buffer: List[Batch] = []
        self.is_eof = False

        self.input_sz = None
        self.output_sz = None
        self.prefill = None
        self.decode = None
        self.lat = None

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False) -> Batch:
        if is_profile:
            return payload
            # if payload.is_eof:
            #     return payload
            # # When profiling, we do not block but stream the immediate result for the current batch.
            # self.prefill, self.decode, self.lat = 0, 0, 0
            # self.input_sz = len(payload.data)
            # if self.input_sz == 0:
            #     return payload
            # if self.group_by is not None:
            #     groups = [payload.data[payload.data[self.group_by] == g] for g in payload.data[self.group_by].unique()]
            # else:
            #     groups = [payload.data]
            # results = []
            # for i, grp in enumerate(groups):
            #     prompts = sem_aggr_template(aggr_col=self.aggr_col, instr=self.instr, df=grp)
            #     llm_result = self.llm_client.llm_call(prompts, **self.llm_args)
            #     responses, prefill, decode, lat = _get_llm_resp(llm_result)
            #     print(f"SemanticAggregation LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            #     if len(responses) != 1:
            #         raise ValueError(f"SemanticAggregation expected 1 response, got {len(responses)}")
            #     self.prefill += prefill
            #     self.decode += decode
            #     self.lat += lat
            #     if self.group_by is not None:
            #         print(grp[self.group_by])
            #         results.append((grp[self.group_by].iloc[0], responses[0]))
            #     else:
            #         results.append((None, responses[0]))
            # if self.group_by is not None:
            #     payload.data = payload.data.groupby(self.group_by).first().reset_index()
            # else:
            #     payload.data = payload.data.iloc[:1].copy()  # just take the first row for aggregation without group-by
            # for i, (g, res) in enumerate(results):
            #     if self.group_by is not None:
            #         payload.data.loc[payload.data[self.group_by] == g, self.out_col] = str(res).strip()
            #     else:
            #         payload.data[self.out_col] = str(res).strip()
            # self.output_sz = len(payload.data)
            # output = Batch(idx=payload.idx, data=payload.data)
            # return output

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            batches = [batch.data for batch in self.buffer]
            if len(batches) == 0:
                all_data = pd.DataFrame()
            else:
                all_data = pd.concat(batches)
            if len(all_data) == 0:
                groups = []
            else:
                # apply group-by
                if self.group_by is not None:
                    groups = [all_data[all_data[self.group_by] == g] for g in all_data[self.group_by].unique()]
                else:
                    groups = [all_data]

            results = []
            for i, grp in enumerate(groups):
                prompts = sem_aggr_template(aggr_col=self.aggr_col, instr=self.instr, df=grp)
                llm_result = _run_llm(self.llm_client, prompts, **self.llm_args)
                responses, prefill, decode, lat = _get_llm_resp(llm_result)
                print(f"SemanticAggregation LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
                if len(responses) != 1:
                    raise ValueError(f"SemanticAggregation expected 1 response, got {len(responses)}")
                if self.group_by is not None:
                    results.append((grp[self.group_by].iloc[0], responses[0]))
                else:
                    results.append((None, responses[0]))
            if len(all_data) > 0:
                if self.group_by is not None:
                    all_data = all_data.groupby(self.group_by).first().reset_index()
                else:
                    all_data = all_data.iloc[:1].copy()  # just take the first row for aggregation without group-by
                for i, (g, res) in enumerate(results):
                    if self.group_by is not None:
                        all_data.loc[all_data[self.group_by] == g, self.out_col] = str(res).strip()
                    else:
                        all_data[self.out_col] = str(res).strip()

            output = Batch(idx=[-1], data=all_data, tables=self.tables)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output
        else:
            return None

    def _get_profile(self):
        return {
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }

class SemanticSort(Operator):
    def __init__(self, oid: int, order_by: str, instr: str, desc: bool = False, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.order_by = order_by
        self.instr = instr
        self.desc = desc
        self.__type_id__ = TYPE_ID.SEM_SORT
        self.op_name = kwargs.get('op_name', None)
        self.tables = []

        self.is_eof = False
        self.buffer: List[Batch] = []
        self.llm_client = llm_client
        self.llm_args = llm_args or {}

        self.input_sz = None
        self.output_sz = None
        self.prefill = None
        self.decode = None
        self.lat = None

    @property
    def oid(self) -> int:
        return self._oid

    @trace_operator
    def process(self, payload: Batch, is_profile=False) -> Batch:
        """Implementation based on naive pairwise comparison in https://github.com/lotus-data/lotus"""

        def compare_batch_binary(
            pairs: list[tuple[dict[str, Any], dict[str, Any]]]
        ) -> tuple[list[bool], list[str], int]:
            match_prompts = []
            for doc1, doc2 in pairs:
                match_prompts.extend(sem_sort_template(order_by=self.order_by, instr=self.instr, doc1=doc1, doc2=doc2))
            llm_result = _run_llm(self.llm_client, match_prompts, **self.llm_args)
            responses, prefill, decode, lat = _get_llm_resp(llm_result)
            print(f"SemanticSort LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")
            comparisons = []
            for resp in responses:
                resp_str = str(resp).strip().lower()
                if resp_str.endswith("1") or resp_str.endswith("2"):
                    comparisons.append((not self.desc) ^ resp_str.endswith("1"))
                else:
                    # Just assume correct ordered
                    comparisons.append(True)
            return comparisons, prefill, decode, lat

        if is_profile:
            return payload
            # if payload.is_eof:
            #     return payload
            # N = len(payload.data)
            # self.input_sz = N
            # pairs = []
            # for i in range(N):
            #     for j in range(i + 1, N):
            #         pairs.append((payload.data.iloc[i][self.order_by], payload.data.iloc[j][self.order_by]))

            # comparisons, prefill, decode, lat = compare_batch_binary(pairs)
            # votes = [0] * N
            # idx = 0
            # for i in range(N):
            #     for j in range(i + 1, N):
            #         if comparisons[idx]:
            #             votes[i] += 1
            #         else:
            #             votes[j] += 1
            #         idx += 1

            # indexes = sorted(range(len(votes)), key=lambda i: votes[i], reverse=True)
            # sorted_data = payload.data.iloc[indexes].reset_index(drop=True)
            # self.prefill = prefill
            # self.decode = decode
            # self.lat = lat
            # self.output_sz = len(sorted_data)
            # output = Batch(idx=[-1], data=sorted_data)
            # return output

        if payload.is_eof:
            self.is_eof = True
        else:
            if is_profile is False:
                self.buffer.append(payload)

        if self.is_eof:
            if self.buffer == []:
                all_data = pd.DataFrame()
            else:
                all_data = pd.concat([batch.data for batch in self.buffer])

            N = len(all_data)
            pairs = []
            for i in range(N):
                for j in range(i + 1, N):
                    pairs.append((all_data.iloc[i][self.order_by], all_data.iloc[j][self.order_by]))

            comparisons, prefill, decode, lat = compare_batch_binary(pairs)
            votes = [0] * N
            idx = 0
            for i in range(N):
                for j in range(i + 1, N):
                    if comparisons[idx]:
                        votes[i] += 1
                    else:
                        votes[j] += 1
                    idx += 1

            indexes = sorted(range(len(votes)), key=lambda i: votes[i], reverse=True)
            sorted_data = all_data.iloc[indexes].reset_index(drop=True)
            output = Batch(idx=[-1], data=sorted_data, tables=self.tables)
            if is_profile is False:
                self.buffer.clear()
                self.is_eof = False
            return output
        else:
            return None

    def _get_profile(self):
        return {
            "input_sz": self.input_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }

class SemanticJoin(Operator):
    def __init__(self, oid: int, left_on, right_on, condition: str, llm_client=None, llm_args: Optional[Dict] = None, **kwargs):
        self._oid = oid
        self.dataset = kwargs.get('dataset', None)
        self.left_on = left_on
        self.right_on = right_on
        self.condition = condition
        self.__type_id__ = TYPE_ID.SEM_JOIN
        self.op_name = kwargs.get('op_name', None)

        self.tables = [None, None]

        self.left_buffer: Optional[Batch] = None
        self.right_buffer: List[Batch] = []
        self.right_cursor = 0
        self.right_eof = False

        self.backup = {}    # Join will need to restore previous exec state when profile made change

        self.llm_client = llm_client
        self.llm_args = llm_args or {}

        self.left_sel = 0
        self.right_sel = 0
        self.input_left_sz = 0
        self.input_right_sz = 0
        self.output_sz = 0
        self.prefill = 0
        self.decode = 0
        self.lat = 0

        self.plan_idx = kwargs.get('plan_idx', None)

    @property
    def oid(self) -> int:
        return self._oid

    def _backup(self):
        self.backup = {
            'left_buffer': self.left_buffer,
            'right_buffer_len': len(self.right_buffer),
            'right_cursor': self.right_cursor,
            'right_eof': self.right_eof
        }

    def _restore(self):
        self.left_buffer = self.backup.get('left_buffer', None)
        right_buffer_len = self.backup.get('right_buffer_len', 0)
        self.right_buffer = self.right_buffer[:right_buffer_len]
        self.right_cursor = self.backup.get('right_cursor', 0)
        self.right_eof = self.backup.get('right_eof', False)

    def _which_table(self, tab, tables: List[str]):
        idx = None
        for i, t in enumerate(tables):
            if tab == t:
                idx = i
                break
        assert idx is not None, f"SemanticJoin operator cannot find the table for table {tab} in tables {tables}"
        return idx

    @trace_operator
    @__config_join__()
    def process(
        self,
        payload: Batch,
        side: str,
        is_profile=False,
        upstream_ckpt=None,
        **kwargs
        ) -> Optional[Batch]:
        """NOTE We do not consider the case where two tables have the same column names here, for simplicity. In current implementation, only one of the two columns will be kept."""

        left_stream, right_stream = upstream_ckpt if upstream_ckpt is not None else ([], [])
        if side == 'left':
            if payload.is_eof:
                # No more left batch will arrive, i.e. join is completed.
                return Batch(idx=[-1], data=pd.DataFrame(), is_eof=True)

            # Checkpoint stream op
            compl = True
            for op in left_stream[::-1]:
                if op.oid in payload.fingerprint:
                    continue
                else:
                    compl = False
                    compl_payload = op.process(payload, is_profile=is_profile)
            # Directly cover the left buffer
            if is_profile is False:
                self.left_buffer = payload if compl else compl_payload
            else:
                self.left_buffer = payload
            # Here the operator should wait for the next right batch to be submitted, so it should not emit any output.
            return None
        elif side == 'right':
            if self.right_eof is False:
                self.right_buffer.append(payload)
                self.right_cursor += 1

            if payload.is_eof:
                self.right_eof = True
                self.right_cursor = 0
                return None
            assert self.left_buffer is not None, "Right batch arrives before left batch for join operator"
            # Emit immediately
            if self.right_eof:
                data = self.right_buffer[self.right_cursor].data
                r_idx = self.right_buffer[self.right_cursor].idx
                right_copy, restored_right = _replay_batch(
                    self.right_buffer[self.right_cursor], right_stream)
                self.right_cursor += 1
            else:
                data = payload.data
                r_idx = payload.idx
                right_copy, restored_right = _replay_batch(payload, right_stream)
            # logger.info(f"Right batch {right_copy.data}")
            left_copy, restored_left = _replay_batch(self.left_buffer, left_stream)
            compl = True
            for op in left_stream[::-1]:
                if op.oid in left_copy.fingerprint:
                    continue
                else:
                    logger.info(f"SemanticJoin operator needs to process left stream op {op.oid} for right batch {left_copy.idx} since it's not in the fingerprint {left_copy.fingerprint}")
                    compl = False
                    left_copy = op.process(left_copy, is_profile=is_profile)
            if (compl is False or restored_left) and is_profile is False:
                self.left_buffer = left_copy

            compl = True
            for op in right_stream[::-1]:
                if op.oid in right_copy.fingerprint:
                    continue
                else:
                    logger.info(f"SemanticJoin operator needs to process right stream op {op.oid} for right batch {right_copy.idx} since it's not in the fingerprint {right_copy.fingerprint}")
                    compl = False
                    right_copy = op.process(right_copy, is_profile=is_profile)
                    if hasattr(op, 'active'):
                        op.active = False
            if (compl is False or restored_right) and is_profile is False:
                self.right_buffer[self.right_cursor-1] = right_copy

            left_values = left_copy.data[self.left_on].tolist() if self.left_on in left_copy.data else []
            right_values = right_copy.data[self.right_on].tolist() if self.right_on in right_copy.data else []
            if not kwargs['use_config'] and 'truncate_input' in kwargs:
                kwargs['truncate_input'](left_copy.data, self.left_on, TYPE_ID.SEM_JOIN)
                kwargs['truncate_input'](right_copy.data, self.right_on, TYPE_ID.SEM_JOIN)

            if kwargs['use_config']:
                left_sel = kwargs['left_sel']
                right_sel = kwargs['right_sel']
                left_in = kwargs['left_in']
                right_in = kwargs['right_in']
                config_len = kwargs['helper_func']

                l_data = left_copy.data
                config_len(l_data, self.left_on, left_in)
                r_data = right_copy.data
                config_len(r_data, self.right_on, right_in)

                left_source = self._which_table(self.tables[0], left_copy.tables)
                right_source = self._which_table(self.tables[1], right_copy.tables)
                l_chosen = configured_source_rows(
                    self.dataset, self.plan_idx, self.op_name,
                    left_copy.tables[left_source], left_copy.idx[left_source],
                    len(left_copy.bitmap[left_source]), left_sel)
                r_chosen = configured_source_rows(
                    self.dataset, self.plan_idx, self.op_name,
                    right_copy.tables[right_source], right_copy.idx[right_source],
                    len(right_copy.bitmap[right_source]), right_sel)
                config_map = []
                for left_meta in left_copy.meta:
                    for right_meta in right_copy.meta:
                        if (left_meta[left_source] in l_chosen
                                and right_meta[right_source] in r_chosen):
                            config_map.append(True)
                        else:
                            config_map.append(False)
                print(config_map)

            # Pairwise join
            idx = 0
            all_prompts = []
            all_config_true = []
            for _, left_row in left_copy.data.iterrows():
                for _, right_row in right_copy.data.iterrows():
                    prompts = sem_join_template(instr=self.condition, doc1=str(left_row[self.left_on]), doc2=str(right_row[self.right_on]), config=config_map[idx] if kwargs['use_config'] else None)
                    all_prompts.extend(prompts)
                    all_config_true.append(('True' if config_map[idx] else 'False') if kwargs['use_config'] else None)
                    idx += 1

            if is_profile and hasattr(self.llm_client, 'set_pair_shape'):
                self.llm_client.set_pair_shape(len(left_copy.data), len(right_copy.data))
            responses, prefill, decode, lat = _get_llm_resp(_run_llm(self.llm_client, all_prompts, **self.llm_args))
            print(f"SemanticJoin LLM call - input_tokens: {prefill}, output_tokens: {decode}, latency_sec: {lat:.2f}")

            if kwargs['use_config']:
                responses = all_config_true
                print(responses)

            joined_data = []
            left_sz, right_sz = len(left_copy.data), len(right_copy.data)
            left_mask, right_mask = [0] * len(left_copy.data), [0] * len(right_copy.data)
            # print(left_copy.meta, right_copy.meta)
            new_meta = []
            pair_keep = []
            idx = 0
            _l, _r = 0, 0
            # print(left_copy.idx, self.right_cursor-1)
            l_tmp, r_tmp = deepcopy(left_copy), deepcopy(right_copy)
            for _, left_row in l_tmp.data.iterrows():
                for _, right_row in r_tmp.data.iterrows():
                    resp_str = str(responses[idx]).strip().lower()
                    keep_pair = resp_str.startswith("true")
                    pair_keep.append(keep_pair)
                    if keep_pair:
                        # print(_l, _r)
                        joined_data.append({**left_row, **right_row,
                                            self.left_on: left_values[_l],
                                            self.right_on: right_values[_r]})
                        left_mask[_l] = 1
                        right_mask[_r] = 1
                        new_meta.append(left_copy.meta[_l] + right_copy.meta[_r])
                    idx += 1
                    _r += 1
                _l += 1
                _r = 0

            # l_tmp._update_bm(
            #     l_tmp.bitmap,
            #     left_mask,
            #     self._which_table(tab=self.tables[0], tables=l_tmp.tables)
            # )
            # r_tmp._update_bm(
            #     r_tmp.bitmap,
            #     right_mask,
            #     self._which_table(tab=self.tables[1], tables=r_tmp.tables)
            # )
            new_tables, new_bm = _merge(l_tmp.tables, l_tmp.bitmap, r_tmp.tables, r_tmp.bitmap)
            joined_data = pd.DataFrame(joined_data)

            if joined_data.empty:
                filtered_left_sz = 0
                filtered_right_sz = 0
            else:
                filtered_left_sz = len(joined_data[self.left_on].unique())
                filtered_right_sz = len(joined_data[self.right_on].unique())
            self.left_sel = filtered_left_sz / left_sz if left_sz > 0 else 0
            self.right_sel = filtered_right_sz / right_sz if right_sz > 0 else 0
            self.input_left_sz = left_sz
            self.input_right_sz = right_sz
            self.output_sz = len(joined_data)
            self.prefill = prefill
            self.decode = decode
            self.lat = lat

            ret = Batch(idx=l_tmp.idx + r_tmp.idx, data=joined_data)
            ret.bitmap = new_bm
            ret.tables = new_tables
            ret.meta = new_meta
            ret.fingerprint = l_tmp.fingerprint + r_tmp.fingerprint + [self.oid]
            ret._sync_bitmap()
            if is_profile or tracing_enabled():
                domain = ProfileMask.from_batch(l_tmp).cross(ProfileMask.from_batch(r_tmp))
                matches = set(ProfileMask.from_batch(ret).rows)
                mask = ProfileMask(domain.tables, domain.rows,
                                   tuple(row in matches for row in domain.rows))
                if is_profile:
                    mask = mask.project(self.tables)
                    record_profile(self, mask, work=left_sz * right_sz)
                observe_mask(self, mask, input_mask=domain, output_mask=ProfileMask.from_batch(ret))
            return ret

    def _get_profile(self):
        if hasattr(self, "_profile_sample"):
            return self._profile_sample
        return {
            "left_sel": self.left_sel,
            "right_sel": self.right_sel,
            "input_left_sz": self.input_left_sz,
            "input_right_sz": self.input_right_sz,
            "output_sz": self.output_sz,
            "prefill": self.prefill,
            "decode": self.decode,
            "lat": self.lat
        }
