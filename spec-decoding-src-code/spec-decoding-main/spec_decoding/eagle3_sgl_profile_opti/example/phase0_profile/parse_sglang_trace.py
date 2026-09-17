#!/usr/bin/env python3
"""对SGLang timeline的分析，聚合 SGLang chrome trace（torch.profiler 导出）：按 GPU kernel / CPU op 名汇总耗时与次数。"""
import gzip
import json
import sys
from collections import defaultdict

path = sys.argv[1]
with gzip.open(path, "rt") as f:
    data = json.load(f)
events = data.get("traceEvents", data if isinstance(data, list) else [])

kern = defaultdict(lambda: [0.0, 0])   # name -> [dur_us, count]
cpu = defaultdict(lambda: [0.0, 0])
memcpy_count = 0
sync_count = 0
for e in events:
    if e.get("ph") != "X":
        continue
    cat = e.get("cat", "")
    name = e.get("name", "")
    dur = float(e.get("dur", 0) or 0)
    if cat in ("kernel", "Kernel", "gpu_op"):
        kern[name][0] += dur
        kern[name][1] += 1
        if "Memcpy" in name or "memcpy" in name:
            memcpy_count += 1
    elif cat in ("cpu_op", "user_annotation", "runtime", "cuda_runtime"):
        cpu[name][0] += dur
        cpu[name][1] += 1
        if "Synchronize" in name or "synchronize" in name:
            sync_count += 1

def show(title, d, n=22):
    tot = sum(v[0] for v in d.values()) or 1.0
    print(f"\n===== {title} (total {tot/1e3:.1f} ms across {len(d)} names) =====")
    print(f"{'name':<62}{'ms':>10}{'%':>7}{'calls':>9}")
    for name, (dur, cnt) in sorted(d.items(), key=lambda x: -x[1][0])[:n]:
        print(f"{name[:60]:<62}{dur/1e3:>10.2f}{100*dur/tot:>6.1f}%{cnt:>9}")

show("SGLang GPU kernels (self device time)", kern)
show("SGLang CPU ops / runtime", cpu)
print(f"\nGPU memcpy kernel events: {memcpy_count}")
print(f"CPU *Synchronize events: {sync_count}")
print(f"total trace events: {len(events)}")
