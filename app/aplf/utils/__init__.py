import time

class Timer:
    start = 0.
    end = 0.
    interval = 0.
    def __enter__(self) -> None:
        self.start = time.clock()
        self.interval = 0.

    def __exit__(self, *args) -> None: # type: ignore
        self.end = time.clock()
        self.interval = self.end - self.start
