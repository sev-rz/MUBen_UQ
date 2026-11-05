#!/usr/bin/env bash
# Point the stage scripts at your own working directory.
# The scripts were run on the MBZUAI cluster and contain the absolute path below.
# Usage: scripts/set_work_dir.sh /your/path/to/work
set -euo pipefail
OLD='/l/users/sevinch.zoda/schub/projects/muben-laplace/work'
NEW="${1:?usage: $0 /your/path/to/work}"
cd "$(dirname "$0")/.."
grep -rlI --include='*.py' --include='*.sh' -F "$OLD" stages | while read -r f; do
  sed -i.bak "s#${OLD}#${NEW}#g" "$f" && rm -f "$f.bak"
done
echo "stage scripts now use $NEW (protocols and result files are left unchanged so their hashes stay valid)"
