"""Verify the portable package's versioned files from any working directory."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text())
for rel,expected in manifest['files'].items():
 path=root/rel
 actual=hashlib.sha256(path.read_bytes()).hexdigest()
 if actual!=expected:raise SystemExit('HASH MISMATCH: '+rel)
print('Verified',len(manifest['files']),'files.')
