#!/usr/bin/env python3
"""Compare versioned workloads without pooling distinct physical strategies."""
import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict

MATCH = ('schema', 'workload_version', 'fixture_version', 'timing_mode', 'threads', 'warmups', 'target_ns', 'comparison_kind')
PHYSICAL = ('configuration', 'physical_work')
REQUIRED = set(MATCH + PHYSICAL + ('case', 'logical_id', 'actual_kernel', 'implementation', 'status', 'ns_per_op', 'comparison_kind'))


def read(path):
    groups = defaultdict(list)
    with open(path, newline='') as source:
        reader = csv.DictReader(source)
        if not REQUIRED.issubset(reader.fieldnames or ()):
            raise ValueError(f'{path}: incompatible or empty benchmark CSV')
        for row in reader:
            if None in row or any(row[key] is None or row[key] == '' for key in REQUIRED - {'ns_per_op'}):
                raise ValueError(f'{path}: malformed benchmark CSV')
            if row['schema'] != '4':
                raise ValueError(f'{path}: unsupported schema; capture a fresh baseline')
            if row['status'] != 'ok':
                continue
            value = float(row['ns_per_op'])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{path}: invalid sample')
            # Keep implementations and physical strategies distinct even in logical mode.
            key = tuple(row[k] for k in ('logical_id',) + MATCH + PHYSICAL + ('implementation', 'actual_kernel', 'comparison_kind'))
            groups[key].append(row)
    if not groups:
        raise ValueError(f'{path}: no successful measurements')
    return list(groups.values())


def compatible(a, b, mode):
    if a['logical_id'] != b['logical_id'] or any(a[k] != b[k] for k in MATCH):
        return False
    if mode == 'fixed':
        return all(a[k] == b[k] for k in PHYSICAL) and a['configuration'] != 'policy-v1'
    # A raw call cannot represent a logical block; neither can format-specific layout maintenance.
    if a['timing_mode'] in ('raw-tile', 'packing-only', 'layout-only', 'vendor-arithmetic'):
        return all(a[k] == b[k] for k in PHYSICAL)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-compatible', action='store_true')
    parser.add_argument('--mode', choices=('fixed', 'logical'), required=True)
    parser.add_argument('--timing', action='append', help='include only this timing boundary; may be repeated')
    parser.add_argument('files', nargs='+')
    args = parser.parse_args()
    if len(args.files) < 2:
        parser.error('provide a baseline and at least one candidate')
    def selected(path):
        return [group for group in read(path)
                if (not args.timing or group[0]['timing_mode'] in args.timing)
                and (args.mode != 'fixed' or group[0]['configuration'] != 'policy-v1')]

    base = selected(args.files[0])
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(('candidate', 'logical_id', 'base_case', 'candidate_case', 'base_configuration', 'candidate_configuration', 'base_physical_work', 'candidate_physical_work', 'base_kernel', 'candidate_kernel', 'base_median_ns', 'candidate_median_ns', 'candidate_min_ns', 'candidate_max_ns', 'base_over_candidate'))
    incompatible = False
    for path in args.files[1:]:
        candidate = selected(path)
        joined = set()
        for left_index, left in enumerate(base):
            for right_index, right in enumerate(candidate):
                a, b = left[0], right[0]
                if not compatible(a, b, args.mode):
                    continue
                x = [float(row['ns_per_op']) for row in left]
                y = [float(row['ns_per_op']) for row in right]
                writer.writerow((path, a['logical_id'], a['case'], b['case'], a['configuration'], b['configuration'], a['physical_work'], b['physical_work'], a['actual_kernel'], b['actual_kernel'], statistics.median(x), statistics.median(y), min(y), max(y), statistics.median(x) / statistics.median(y)))
                joined.add(('base', left_index)); joined.add(('candidate', right_index))
        # A completely unmatched report must not succeed silently. Each strategy needs a compatible peer
        # when the mathematical workload appears on both sides; unsupported rows are not measurements.
        common = {g[0]['logical_id'] for g in base} & {g[0]['logical_id'] for g in candidate}
        if not joined:
            print(f'incompatible case=no compatible measurements in {path}', file=sys.stderr)
            incompatible = True
        for side, groups in (('base', base), ('candidate', candidate)):
            for index, group in enumerate(groups):
                if group[0]['logical_id'] in common and (side, index) not in joined:
                    print(f"incompatible case={group[0]['case']} side={side} mode={args.mode}", file=sys.stderr)
                    incompatible = True
    return 1 if args.require_compatible and incompatible else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
