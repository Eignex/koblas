function expand(list, cpus,    parts, bounds, n, i, cpu, last) {
    n = split(list, parts, ",")
    for (i = 1; i <= n; i++) {
        split(parts[i], bounds, "-")
        last = bounds[2] == "" ? bounds[1] : bounds[2]
        for (cpu = bounds[1]; cpu <= last; cpu++) cpus[cpu] = 1
    }
}

function load(cpu,    total, idle, busy) {
    if (!(cpu in beforeTotal) || !(cpu in afterTotal)) return 1
    total = afterTotal[cpu] - beforeTotal[cpu]
    idle = afterIdle[cpu] - beforeIdle[cpu]
    if (total <= 0) return 1
    busy = 1 - idle / total
    return busy < 0 ? 0 : (busy > 1 ? 1 : busy)
}

function score(cpu,    path, list, siblings, sibling, busy, maximum) {
    path = topology "/cpu" cpu "/topology/thread_siblings_list"
    if ((getline list < path) <= 0) list = cpu
    close(path)
    expand(list, siblings)
    # Busy SMT siblings count even when they are outside the allowed CPU set.
    maximum = 0
    for (sibling in siblings) {
        busy = load(sibling)
        if (busy > maximum) maximum = busy
    }
    return int(maximum * 20)
}

BEGIN {
    srand(seed % 2147483647)
    expand(allowed, candidates)
    minimum = 21
}

/^cpu[0-9]+[[:space:]]/ {
    cpu = substr($1, 4)
    total = 0
    # Guest counters are already included in user/nice; do not count them twice.
    for (field = 2; field <= 9; field++) total += $field
    if (FILENAME == ARGV[1]) {
        beforeTotal[cpu] = total
        beforeIdle[cpu] = $5 + $6
    } else {
        afterTotal[cpu] = total
        afterIdle[cpu] = $5 + $6
    }
}

END {
    for (cpu in candidates) {
        bucket = score(cpu)
        if (bucket < minimum) {
            minimum = bucket
            selected = cpu
            ties = 1
        } else if (bucket == minimum && rand() < 1 / ++ties) selected = cpu
    }
    if (minimum == 21) exit 1
    print selected
}
