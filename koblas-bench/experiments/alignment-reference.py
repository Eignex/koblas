#!/usr/bin/env python3
"""Compare owned JVM reductions with native oneMKL at fixed and matched buffer alignments."""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = 'intel-oneapi-mkl-core-2026.1-2026.1.0-236_amd64.deb'
PACKAGE_SHA = '3b230c02fc0ac44f8f71062c20cd883b37c6c1e00940dc0a7f13b02d9c47bd93'
HEADER = ('round,sample,operation,arm,offset,iterations,elapsed_ns,ns_per_call,heap_a_before,'
          'heap_a_after,heap_b_before,heap_b_after,native_a,native_b,heap_y_before,heap_y_after,'
          'native_y,checksum,bytes_per_call,gc_events,prefix_a,prefix_b')


def run(command, output, environment):
    with output.open('w') as stream:
        subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--java', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--layout', action='append', choices=['zgc', 'serial', 'g1'])
    parser.add_argument('--mkl-root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    java = args.java.resolve()
    environment = os.environ.copy()
    for name in ['JAVA_TOOL_OPTIONS', 'JDK_JAVA_OPTIONS', '_JAVA_OPTIONS']:
        environment.pop(name, None)
    layouts = args.layout or ['zgc', 'serial']
    run([str(java), '-version'], output / 'java-version.log', environment)
    version_log = (output / 'java-version.log').read_text()
    assert re.search(r'version \"25\.', version_log), 'The alignment reference requires JDK 25'
    staged = ROOT / 'koblas-bench/build/alignment-diagnostic-runtime'
    runtime = output / 'runtime'
    shutil.copytree(staged, runtime)
    entries = [runtime / Path(item).name for item in
               (ROOT / 'koblas-bench/build/alignment-diagnostic-classpath.txt').read_text().strip().split(os.pathsep)]
    assert all(entry.exists() for entry in entries)
    jars = [entry for entry in entries if entry.name.endswith('koblas-jvm-SNAPSHOT.jar')]
    assert len(jars) == 1
    classpath = os.pathsep.join(map(str, entries))
    manifest = output / 'classpath.txt'
    manifest.write_text(classpath)
    for name in ['KotlinAlignmentComparison.java']:
        shutil.copy2(ROOT / 'koblas-bench/experiments' / name, output / name)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    classes = output / 'classes'
    classes.mkdir()
    run([str(java.with_name('javac')), '--add-modules=jdk.incubator.vector', '-d', str(classes),
         str(output / 'KotlinAlignmentComparison.java')], output / 'compile-java.log', environment)
    if args.mkl_root:
        mkl = args.mkl_root.resolve()
    else:
        scratch = tempfile.TemporaryDirectory(prefix='alignment-reference-', dir=ROOT / 'koblas-bench/build')
        vendor_scratch = Path(scratch.name)
        package = vendor_scratch / PACKAGE
        urllib.request.urlretrieve('https://apt.repos.intel.com/oneapi/pool/main/' + PACKAGE, package)
        assert hashlib.sha256(package.read_bytes()).hexdigest() == PACKAGE_SHA
        run(['dpkg-deb', '--extract', str(package), str(vendor_scratch / 'vendors')], output / 'extract-mkl.log', environment)
        mkl = vendor_scratch / 'vendors/opt/intel/oneapi/mkl/2026.1'
    assert (mkl / 'lib/libmkl_rt.so.3').is_file()
    environment.update(MKLROOT=str(mkl), LD_LIBRARY_PATH=str(mkl / 'lib') + ':' + environment.get('LD_LIBRARY_PATH', ''),
                       MKL_THREADING_LAYER='SEQUENTIAL')
    cpu = min(os.sched_getaffinity(0))
    paths = [java, manifest, output / 'KotlinAlignmentComparison.java',
             output / Path(__file__).name, mkl / 'lib/libmkl_rt.so.3']
    paths += [f for directory in [runtime, classes] for f in directory.rglob('*') if f.is_file()]

    def hashes():
        return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    original = hashes()
    plan = dict(status='incomplete', kind='smoke' if args.smoke else 'raw JVM/native fixed-alignment diagnostic',
                cpu=cpu, forks=1 if args.smoke else 2, rounds=1 if args.smoke else 3,
                samples=1 if args.smoke else 5, length=65536, offsets=list(range(8)), layouts=layouts,
                input_hashes=original, oneMKL_package=dict(file=PACKAGE, sha256=PACKAGE_SHA),
                vendor_environment={name: environment.get(name) for name in
                                    ['MKL_ENABLE_INSTRUCTIONS', 'MKL_DEBUG_CPU_TYPE', 'MKL_CBWR', 'MKL_THREADING_LAYER']},
                notes='Native address snapshots occur outside arithmetic. Native buffers are explicitly 64-byte aligned or match the unpeeled heap window. No best-offset selection; not the public Kotlin/Native capture.')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    allrows = []
    for layout in layouts:
        flags = {'zgc': ['-XX:+UseZGC'], 'serial': ['-XX:+UseSerialGC'],
                 'g1': ['-XX:+UseG1GC', '-XX:G1HeapRegionSize=1m']}[layout]
        for fork in range(1, plan['forks'] + 1):
            command = ['taskset', '-c', str(cpu), str(java), '-Xms1g', '-Xmx1g', *flags,
                       '--add-modules=jdk.incubator.vector', '--enable-native-access=ALL-UNNAMED']
            if args.smoke:
                command += ['-Dtrial.iterations=10', '-Dtrial.samples=1', '-Dtrial.rounds=1', '-Dtrial.warmupNanos=1000000']
            command += ['-cp', str(classes), 'KotlinAlignmentComparison', str(output), 'candidate=' + str(jars[0]), str(manifest)]
            name = f'{layout}-fork-{fork}'
            (output / (name + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
            print('START', name, flush=True)
            run(command, output / (name + '.log'), environment)
            text = (output / (name + '.log')).read_text()
            assert 'address_probe=libc/memchr' in text
            assert 'oneMKL configured max threads=1' in text and 'alignment_policy arm=candidate' in text
            assert 'version=25.' in text and 'oneMKL version=' in text and 'Version 2026.1' in text
            assert re.search(r'preferred_lanes=(2|4|8|16|32|64)\b', text)
            lines = [line for line in text.splitlines() if len(line.split(',')) == len(HEADER.split(','))
                     and line.split(',')[0].isdigit()]
            rows = list(csv.DictReader([HEADER] + lines))
            assert len(rows) == plan['rounds'] * plan['samples'] * 8 * 3 * 3
            for row in rows:
                row.update(layout=layout, fork=fork)
                if row['arm'] == 'onemkl-aligned':
                    assert int(row['native_a']) % 64 == 0 and int(row['native_b']) % 64 == 0
                if row['arm'] == 'onemkl-matched':
                    assert row['native_a'] == row['heap_a_before'] and row['native_b'] == row['heap_b_before']
            allrows += rows
            assert hashes() == original
            print('DONE', name, flush=True)
    with (output / 'samples.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(allrows[0]))
        writer.writeheader()
        writer.writerows(allrows)
    groups = collections.defaultdict(list)
    for row in allrows:
        groups[(row['layout'], row['operation'], row['arm'], int(row['offset']), int(row['round']))].append(float(row['ns_per_call']))
    rounds = collections.defaultdict(list)
    for (layout, op, arm, offset, _), values in groups.items():
        assert len(values) == plan['forks'] * plan['samples']
        rounds[(layout, op, arm, offset)].append(statistics.median(values))
    summary = [dict(layout=k[0], operation=k[1], arm=k[2], offset=k[3], median_ns=statistics.median(v),
                    round_medians_ns=v, min_round_ns=min(v), max_round_ns=max(v)) for k, v in rounds.items()]
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plan.update(status='complete', samples_total=len(allrows), timed_gc_events=sum(int(row['gc_events']) for row in allrows),
                moving_samples=sum(row['heap_a_before'] != row['heap_a_after'] or row['heap_b_before'] != row['heap_b_after'] for row in allrows),
                nonzero_allocation_samples=[row for row in allrows if float(row['bytes_per_call']) != 0])
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps({k: v for k, v in plan.items() if k not in ['input_hashes', 'nonzero_allocation_samples']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
