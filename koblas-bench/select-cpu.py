#!/usr/bin/env python3
"""Choose a lightly loaded allowed physical core without reserving a worker."""

import os
from pathlib import Path
import random
import time


def cpu_times():
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        fields = line.split()
        if fields[0].startswith('cpu') and fields[0][3:].isdigit():
            # Guest time is already included in user/nice; do not count it twice.
            values = [int(value) for value in fields[1:9]]
            result[int(fields[0][3:])] = (sum(values), values[3] + values[4])
    return result


def siblings(cpu):
    try:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text()
    except OSError:
        return {cpu}
    result = set()
    for part in topology.strip().split(','):
        bounds = [int(value) for value in part.split('-')]
        result.update(range(bounds[0], bounds[-1] + 1))
    return result


def choose_cpu(allowed, before, after, topology, rng):
    def load(cpu):
        if cpu not in before or cpu not in after:
            return 1.0
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        return max(0.0, min(1.0, 1.0 - idle / total)) if total > 0 else 1.0

    # Count busy SMT siblings even when the job cannot itself run on them. Bucket
    # near-idle cores together so simultaneous captures do not all choose CPU 0.
    scores = {cpu: int(max(load(sibling) for sibling in topology(cpu)) * 20) for cpu in allowed}
    minimum = min(scores.values())
    return rng.choice([cpu for cpu in sorted(allowed) if scores[cpu] == minimum])


if __name__ == '__main__':
    allowed = os.sched_getaffinity(0)
    before = cpu_times()
    time.sleep(0.2)
    print(choose_cpu(allowed, before, cpu_times(), siblings, random.SystemRandom()))
