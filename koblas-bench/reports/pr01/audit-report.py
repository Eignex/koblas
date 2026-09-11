import csv, math
from collections import Counter
from pathlib import Path
root=Path('koblas-bench/reports/f342c3192a41a07cf0454d7d4d5324f8289cc01329bcc9c990b986dcdd2f5922')
lines=[]
for run in sorted(p for p in root.iterdir() if p.is_dir()):
 assert (run/'status.txt').read_text().strip()=='complete'
 cases=[l for l in (run/'cases.txt').read_text().splitlines() if l and not l.startswith('#')]
 master=set(l for l in Path('koblas-bench/cases.txt').read_text().splitlines() if l and not l.startswith('#'))
 assert set(cases)<=master
 source=run.name.split('-')[-1]
 lines.append(f'{run.name}: {len(cases)} authoritative case lines; source.patch bytes={(run/"source.patch").stat().st_size}')
 for file in list(run.glob('*.csv'))+[run/'vendor/openblas.csv']:
  with file.open() as f:rows=list(csv.DictReader(f))
  assert all(None not in r and all(v is not None for v in r.values()) for r in rows)
  assert {r['case'] for r in rows}==set(cases)
  assert {r['schema'] for r in rows}=={'4'}
  assert {r['workload_version'] for r in rows}=={'5'}
  assert {r['fixture_version'] for r in rows}=={'2'}
  assert all(r['source_commit'].startswith(source) for r in rows)
  counts=Counter(r['case'] for r in rows if r['status']=='ok')
  assert set(counts.values())=={5}
  for r in rows:
   if r['status']=='ok':
    ns=float(r['ns_per_op']);op=int(r['operations']);elapsed=int(r['elapsed_ns'])
    assert math.isfinite(ns) and ns>0 and op>0 and elapsed>0
    assert abs(ns-elapsed/op)<=max(1.0/op,1e-8*ns)
    assert r['actual_kernel']!='unavailable'
   else:
    assert r['status']=='unsupported' and r['ns_per_op']=='' and r['sample']=='0'
  lines.append(f'  {file.relative_to(run)}: {len(rows)} rows; {len(counts)} supported cases; {sum(r["status"]=="unsupported" for r in rows)} unsupported; five valid samples per supported case')
lines.append('All report case sets, metric fields, schema/workload/fixture versions, source SHAs and timing arithmetic passed.')
Path('koblas-bench/reports/pr01/report-audit.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
