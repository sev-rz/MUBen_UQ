"""Freeze PROTOCOL_v2.md from PROTOCOL_v2_TEMPLATE.md. Refuses unless: no full-run output exists yet (results_v2/), check_inputs_v2.py exits 0,
the validation-half self-test (selftest_v2/) is newer than every prepared validation folder, and CHECKS_v2.txt (the written outcome of the section-5 checks) exists.
Writes PROTOCOL_v2.md and PROTOCOL_v2.sha256 (sha256 of PROTOCOL_v2.md); never overwrites an existing PROTOCOL_v2.md.
  python freeze_v2.py"""
import sys, glob, hashlib, subprocess, datetime as dt
from pathlib import Path
C = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (C / 'PROTOCOL_v2.md').exists(), 'PROTOCOL_v2.md already frozen'
assert not (C / 'results_v2').exists(), 'results_v2/ exists: the full run has already started, too late to freeze'
r = subprocess.run([sys.executable, str(C / 'check_inputs_v2.py')], capture_output=True, text=True); print(r.stdout)
assert r.returncode == 0, 'check_inputs_v2.py failed'
newest = max(Path(p).stat().st_mtime for p in glob.glob(str(C / 'preds' / '*' / '*' / '*' / 'DONE')))
st = [C / 'selftest_v2' / 'conformal_regression.csv', C / 'selftest_v2' / 'conformal_classification.csv']
assert all(p.exists() and p.stat().st_mtime > newest for p in st), 'self-test missing or older than the newest prepared folder'
assert (C / 'CHECKS_v2.txt').exists(), 'write CHECKS_v2.txt first'
files = ['conformal_v2.py', 'check_inputs_v2.py', 'conformal_summary.py', 'valid_preds.py', 'valid_preds_gpu.py', 'valid_preds_gin.py', 'valid_preds_laplace.py', 'PROTOCOL.md']
dig = '\n'.join(f'- {f}: {sha(C / f)}' for f in files)
txt = (C / 'PROTOCOL_v2_TEMPLATE.md').read_text().replace('{{DATE}}', dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%MZ')).replace('{{CHECKS}}', (C / 'CHECKS_v2.txt').read_text().strip()).replace('{{DIGESTS}}', dig)
assert '{{' not in txt
(C / 'PROTOCOL_v2.md').write_text(txt); (C / 'PROTOCOL_v2.sha256').write_text(f"{sha(C / 'PROTOCOL_v2.md')}  PROTOCOL_v2.md\n")
print('FROZEN PROTOCOL_v2.md', sha(C / 'PROTOCOL_v2.md'))
