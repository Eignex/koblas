"""Export compact, reviewable reports without modifying the raw capture."""

import argparse
import csv
from datetime import datetime
import math
from pathlib import Path
import shutil
import statistics


IMPLEMENTATIONS = ('jvm-scalar', 'jvm-c', 'jvm-simd', 'native', 'openblas', 'accelerate', 'onemkl')


def number(value):
    return format(value, '.17g')


def summarize_benchmark(source, destination):
    with source.open(newline='') as stream:
        rows = list(csv.reader(stream))
    if len(rows) < 3 or rows[2] != ['sample', 'case_id', 'fork', 'sample', 'operations', 'elapsed_ns', 'ns_per_op']:
        raise ValueError(f'{source}: expected raw benchmark CSV')
    runs, cases, samples, seen = {}, {}, {}, set()
    for row in rows[3:]:
        if not row:
            continue
        if row[0] == 'run':
            if row[1] in runs:
                raise ValueError('duplicate run')
            runs[row[1]] = row
        elif row[0] == 'case':
            if row[1] in cases or row[2] not in runs or row[4] not in ('ok', 'unsupported'):
                raise ValueError('invalid case')
            cases[row[1]] = row
            samples[row[1]] = []
        elif row[0] == 'sample':
            key = tuple(row[1:4])
            if row[1] not in cases or cases[row[1]][4] != 'ok' or key in seen:
                raise ValueError('invalid or duplicate sample')
            value = float(row[6])
            if not math.isfinite(value) or value <= 0 or any(int(v) <= 0 for v in row[2:6]):
                raise ValueError('invalid sample value')
            seen.add(key)
            samples[row[1]].append((row[2], value))
        else:
            raise ValueError('unknown benchmark record')
    summaries = []
    for key, case in cases.items():
        values = [value for _, value in samples[key]]
        forks = len({fork for fork, _ in samples[key]})
        if case[4] == 'ok' and (not values or forks != int(runs[case[2]][-1])):
            raise ValueError('missing measured samples or forks')
        stats = [number(statistics.median(values)), number(min(values)), number(max(values))] if values else ['', '', '']
        summaries.append(case + [len(values), forks] + stats)
    with destination.open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(rows[0])
        writer.writerow(rows[1] + ['samples', 'forks', 'median_ns', 'min_ns', 'max_ns'])
        writer.writerows(runs.values())
        writer.writerows(summaries)


def summarize_cpu(source, metadata, destination):
    fields = dict(line.split('=', 1) for line in metadata.splitlines() if '=' in line)
    start = datetime.fromisoformat(fields['started_at'].replace('Z', '+00:00'))

    def elapsed(key):
        return (datetime.fromisoformat(fields[key].replace('Z', '+00:00')) - start).total_seconds()

    boundaries = [('baseline', 0.0)]
    boundaries += [(name, elapsed(name + '_started_at')) for name in
                   ('jvm-scalar', 'jvm-c', 'jvm-simd', 'native', 'vendors')]
    boundaries.append(('end', elapsed('completed_at')))
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    with destination.open('x', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['phase', 'start_s', 'end_s', 'readings', 'available_readings',
                         'mean_cpu_percent', 'median_cpu_percent', 'min_cpu_percent', 'max_cpu_percent'])
        for (phase, low), (_, high) in zip(boundaries, boundaries[1:]):
            if high < low:
                raise ValueError('CPU phase timestamps are out of order')
            selected = [row for row in rows if low <= float(row['elapsed_s']) < high]
            values = [float(row['cpu_percent']) for row in selected if row['cpu_percent']]
            if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
                raise ValueError('invalid CPU percentage')
            stats = [format(value, '.6g') for value in
                     (statistics.mean(values), statistics.median(values), min(values), max(values))] if values else [''] * 4
            writer.writerow([phase, f'{low:.3f}', f'{high:.3f}', len(selected), len(values)] + stats)


def summarize_report(source, destination):
    metadata = (source / 'metadata.txt').read_text()
    if not metadata.startswith('status=complete\n'):
        raise ValueError('cannot summarize an incomplete capture')
    benchmarks = [source / f'{name}.csv' for name in IMPLEMENTATIONS if (source / f'{name}.csv').is_file()]
    if not benchmarks:
        raise ValueError('no raw benchmark files')
    destination.mkdir(parents=True, exist_ok=False)
    for benchmark in benchmarks:
        summarize_benchmark(benchmark, destination / f'{benchmark.stem}-summary.csv')
    summarize_cpu(source / 'cpu.csv', metadata, destination / 'cpu-summary.csv')
    shutil.copy2(source / 'metadata.txt', destination / 'metadata.txt')
    if (source / 'source.patch').exists():
        shutil.copy2(source / 'source.patch', destination / 'source.patch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw_directory', type=Path)
    parser.add_argument('output_directory', type=Path, help='new directory; existing reports are never overwritten')
    arguments = parser.parse_args()
    summarize_report(arguments.raw_directory, arguments.output_directory)
