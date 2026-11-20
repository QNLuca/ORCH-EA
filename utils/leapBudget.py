import os
import redis

BUDGET_KEY = "leap:qpu_budget_us"
CONSUMED_KEY = "leap:qpu_consumed_us"
INIT_KEY = "leap:budget_initialized"

def _client():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=0,
        decode_responses=True,
    )

def init_budget_once(total_us: int):
    r = _client()
    if r.set(INIT_KEY, "1", nx=True):  # solo se non esiste
        r.set(BUDGET_KEY, int(total_us))
        r.set(CONSUMED_KEY, 0)
    return get_status()

def can_afford(estimated_us: int, safety: float = 1.25) -> bool:
    r = _client()
    left = int(r.get(BUDGET_KEY) or 0)
    return left >= int(estimated_us * safety)

def charge(actual_us: int):
    if actual_us is None:
        return
    actual_us = int(actual_us)
    r = _client()
    pipe = r.pipeline()
    pipe.incrby(CONSUMED_KEY, actual_us)
    pipe.decrby(BUDGET_KEY, actual_us)
    pipe.execute()

def get_status():
    r = _client()
    return {
        "remaining_us": int(r.get(BUDGET_KEY) or 0),
        "consumed_us": int(r.get(CONSUMED_KEY) or 0),
    }