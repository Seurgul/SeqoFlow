"""Execution-only controls scoped to one subquery pull."""

from contextvars import ContextVar


_full_batch = ContextVar('full_batch_execution', default=False)


def full_batch_execution():
    return _full_batch.get()


def set_full_batch_execution(enabled):
    return _full_batch.set(enabled)


def reset_full_batch_execution(token):
    _full_batch.reset(token)
