from collections import defaultdict
import time


class Profiler:
    def __init__(self):
        self.times = {}
        self.start ={}

    def timer(self,name,enable=True):
        return self.TimerContext(self,name,enable)
    

    class TimerContext:
        def __init__(self,profiler,name,enable):
            self.p = profiler
            self.name = name
            self.enable = enable


        def __enter__(self):
            if self.enable:
                self.p.start[self.name] = time.perf_counter()

        def __exit__(self, exc_type, exc, tb):
            if self.enable:
                elapsed = time.perf_counter()-self.p.start[self.name]
                self.p.times[self.name]=elapsed

    def report(self):
        all_time = sum(self.times.values())

        for name,times in self.times.items():
            print(f"module: {name}, usage: {times:.8f}s, {(times/all_time)*100:.2f}")