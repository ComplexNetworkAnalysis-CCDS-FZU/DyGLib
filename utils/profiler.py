from collections import defaultdict
import time




import time
import json
import numpy as np
import torch
from collections import defaultdict

class Profiler:
    def __init__(self, sync_cuda=True):
        self.times = {}              # 当前这一次 forward 的耗时
        self.start = {}
        self.sync_cuda = sync_cuda

        # 跨 batch 累积
        self._accumulated = defaultdict(list)   # name -> [elapsed_seconds, ...]

        self.enabled = True

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def timer(self, name, enable=True):
        return self.TimerContext(self, name, enable and self.enabled)

    class TimerContext:
        def __init__(self, profiler, name, enable):
            self.p = profiler
            self.name = name
            self.enable = enable

        def __enter__(self):
            if self.enable:
                if self.p.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.p.start[self.name] = time.perf_counter_ns()   # 修复：统一用 perf_counter()

        def __exit__(self, exc_type, exc, tb):
            if self.enable:
                if self.p.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.perf_counter_ns() - self.p.start[self.name]
                self.p.times[self.name] = elapsed
                self.p._accumulated[self.name].append(elapsed)  # 累积

    def report(self):
        all_time = sum(self.times.values())
        if all_time == 0:
            return
        for name, elapsed in self.times.items():
            pct = (elapsed / all_time) * 100
            print(f"module: {name}, \tusage: {elapsed:.8f}ns, \t{pct:.2f}%")
        print(f"{'TOTAL':30s}  {all_time:8.3f}ns\n")

    def summary(self, warmup_ratio=0.1):
        """跳过前 warmup_ratio 的 batch，返回汇总统计"""
        result = {}
        for name, values in self._accumulated.items():
            if len(values) == 0:
                continue
            warmup = max(1, int(len(values) * warmup_ratio))
            if len(values) <= warmup:
                # 数据量不足 warmup 时退化为使用全部数据，避免空数组崩溃
                data = np.array(values)
            else:
                data = np.array(values[warmup:])
            result[name] = {
                'mean_ns':   float(np.mean(data)),
                'std_ns':    float(np.std(data)),
                'min_ns':    float(np.min(data)),
                'max_ns':    float(np.max(data)),
                'total_s':   float(np.sum(data)),
                'n_calls':   int(len(data)),
            }
        return result

    def save(self, filepath):
        with open(f"{filepath}-profiler.json", 'w') as f:
            json.dump(self.summary(), f, indent=2)

    def reset(self):
        self.times.clear()
        self.start.clear()
        self._accumulated.clear()
