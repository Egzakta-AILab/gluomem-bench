"""Portable continuation accounting; campaign-specific operators stay private."""


class NewAttemptBudget:
    """New-journal coordinates over ONE original-total guard, not another budget.

    The unchanged coordinator's journal starts at zero. All guard state, deadlines,
    locks, reserves and real debit storage remain in the original CampaignBudget.
    Only journal-facing call/time coordinates omit the independently bound prior.
    """
    def __init__(self, budget):
        self.outer = budget
        self.max_calls = budget.max_calls - budget.prior_calls
        self.max_seconds = budget.max_seconds - budget.prior_elapsed_seconds

    def __getattr__(self, name):
        return getattr(self.outer, name)

    @property
    def calls(self):
        return self.outer.calls - self.outer.prior_calls

    @calls.setter
    def calls(self, value):
        self.outer.calls = self.outer.prior_calls + value

    @property
    def reserved_calls(self):
        return self.outer.reserved_calls

    @reserved_calls.setter
    def reserved_calls(self, value):
        self.outer.reserved_calls = value

    @property
    def failed(self):
        return self.outer.failed

    @failed.setter
    def failed(self, value):
        self.outer.failed = value

    def snapshot(self):
        s = self.outer.snapshot()
        return {'calls': s['new_calls'], 'elapsed_seconds': s['new_elapsed_seconds']}
