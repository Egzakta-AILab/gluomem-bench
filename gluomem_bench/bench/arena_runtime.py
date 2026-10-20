"""Explicit reservation primitives; callers retain admission and handler ordering."""
from .parallel_campaign import CampaignStopped, JobContext


def reserve_context(budget, job, limiter, refusal):
    """Reserve after caller admission; a refused task has no JobContext."""
    if not budget.reserve(job):
        raise CampaignStopped(refusal)
    return JobContext(job, budget, limiter)


def release_context(context, row):
    """Call in the caller's finally, AFTER handlers and BEFORE status writes."""
    if context is not None:
        row["calls"] = context.calls
        context.budget.release(context)
