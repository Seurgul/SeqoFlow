import logging


logger = logging.getLogger(__name__)

class Memoizer:
    """
    A Memoizer is responsible to keep track of all logical plan instances for later profiling.
    """
    def __init__(self):
        self._memo = []
        self._seen = set()  # tuples of labeled partition position vectors already memoized

    def admit(self, key):
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def add(self, plan):
        self._memo.append(plan)
        logger.debug(f"Added plan to memoizer: \n{plan.print_plan()}")
        return len(self._memo) - 1

    def get(self, plan_id):
        return self._memo[plan_id]

    def clear(self):
        self._memo = []
        self._seen = set()


# The global memoizer instance
def get_memoizer():
    global _global_memoizer
    if '_global_memoizer' not in globals():
        _global_memoizer = Memoizer()
    return _global_memoizer