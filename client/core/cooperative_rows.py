"""Time-sliced iterator driver. No wx dependency, callbacks are injected.

One operation may exceed the budget; the next one always gets a new turn.
Cancellation closes the iterator only outside a running step.
"""
import time


class CooperativeRows:
    def __init__(self, operations, schedule, finished, valid=lambda: True,
                 clock=time.monotonic, budget=0.008, max_ops=32):
        self.operations = iter(operations)
        self.schedule = schedule
        self.finished = finished
        self.valid = valid
        self.clock = clock
        self.budget = budget
        self.max_ops = max_ops
        self.cancelled = False
        self.running = False
        self.done = False

    def cancel(self):
        self.cancelled = True
        if not self.running and not self.done:
            self.done = True
            self.operations.close()

    def step(self):
        if self.done:
            return
        if self.cancelled or not self.valid():
            self.cancel()
            return
        started = self.clock()
        self.running = True
        complete = False
        try:
            for _ in range(self.max_ops):
                try:
                    next(self.operations)
                except StopIteration:
                    complete = True
                    break
                if self.cancelled or self.clock() - started >= self.budget:
                    break
        finally:
            self.running = False
        if self.cancelled:
            self.cancel()
        elif complete:
            self.done = True
            self.finished()
        else:
            self.schedule(self.step)
