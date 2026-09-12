#!/usr/bin/env python3
"""Summarize the retained raw diagnostic and suite samples without rewriting them."""
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
print('## Native engine diagnostic\n')
print('Thread CPU ns/op; medians across three passes and seven samples, grouping measured pointer alignment.\n')
print('| n | Timing | Alignment | Before | After | Median paired speedup | Pass range |')
print('|---:|---|---|---:|---:|---:|---:|')
native = defaultdict(list)
for path in sorted(root.glob('native-*-*.csv')):
    _, arm, pass_id = path.stem.split('-')
    for row in csv.DictReader(path.open()):
        alignment = 'aligned' if row['alignment_bytes'] == '0' else 'misaligned'
        native[int(row['n']), row['mode'], alignment, arm, pass_id].append(float(row['cpu_ns_per_op']))
for n in (64, 256, 4096, 65536):
    for mode in ('arithmetic', 'reset'):
        for alignment in ('aligned', 'misaligned'):
            before = [x for p in ('1','2','3') for x in native[n,mode,alignment,'before',p]]
            after = [x for p in ('1','2','3') for x in native[n,mode,alignment,'after',p]]
            ratios = [st.median(native[n,mode,alignment,'before',p]) / st.median(native[n,mode,alignment,'after',p]) for p in ('1','2','3')]
            print(f'| {n} | {mode} | {alignment} | {st.median(before):.1f} | {st.median(after):.1f} | {st.median(ratios):.2f}x | {min(ratios):.2f}–{max(ratios):.2f}x |')
print('\n## C leaf and vendor diagnostic\n')
print('Thread CPU ns/op; final guarded candidate, medians across seven samples and the three misaligned offsets (8, 16, 24 bytes). Each arm rotates within a process.\n')
print('| n | Timing | Original | Guarded alignment | oneMKL | OpenBLAS |')
print('|---:|---|---:|---:|---:|---:|')
leaf = defaultdict(list)
for row in csv.DictReader((root/'guarded.csv').open()):
    if row['offset_bytes'] != '0':
        leaf[int(row['n']),row['mode'],row['implementation']].append(float(row['cpu_ns_per_op']))
for n in (64,128,256,4096,65536):
    for mode in ('negate','powers-two','reset'):
        values = [st.median(leaf[n,mode,a]) for a in ('baseline','aligned','onemkl','openblas')]
        print('| '+str(n)+' | '+mode+' | '+' | '.join(f'{x:.1f}' for x in values)+' |')
if (root/'suite').exists():
    print('\n## Standard benchmark suite\n')
    print('Elapsed ns/op; medians across three alternating passes, five samples per pass, five warmups and 200 ms targets. Array addresses are uncontrolled in this suite.\n')
    suite = defaultdict(list)
    for path in sorted((root/'suite').glob('p*-*.csv')):
        arm = path.stem.split('-',1)[1]
        ids = {}
        for row in csv.reader(path.open()):
            if row and row[0] == 'case' and row[1] != 'id': ids[row[1]] = row[3]
            elif row and row[0] == 'sample' and row[1] != 'case_id': suite[ids[row[1]],arm].append(float(row[-1]))
    print('| Case | Native before | Native after | oneMKL | OpenBLAS |')
    print('|---|---:|---:|---:|---:|')
    for case in sorted({c for c,a in suite}):
        values = [st.median(suite[case,a]) for a in ('before','native','onemkl','openblas')]
        print('| '+case+' | '+' | '.join(f'{x:.1f}' for x in values)+' |')
