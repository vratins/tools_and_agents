#!/bin/bash
# Prepare only: no package downloads, builds, or scheduler submissions here.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
toolkit=$(cd -- "$here/../.." && pwd)
uv_bin=$(command -v uv)
context="$here/.chtc/build-test"
mkdir -p "$context"
python3 - "$here" "$toolkit" "$uv_bin" "$context" <<'PY'
from pathlib import Path
import shutil, sys
here, toolkit, uv, context = map(Path, sys.argv[1:])
for name in ('smoke.py', 'pyproject.toml', 'build-and-test.sh'):
    shutil.copy2(here / name, context / name)
shutil.copy2(toolkit / 'bin' / 'chtc', context / 'chtc')
def quote(path):
    value = str(path)
    if any(c in value for c in '\n\r\"$,'):
        raise SystemExit('Unsupported character in input path')
    return '"' + value + '"'
# Transfer uv directly from its existing path rather than duplicating 48 MB in home.
inputs = [uv, context / 'chtc', context / 'smoke.py', context / 'pyproject.toml']
(context / 'build-test.sub').write_text('\n'.join([
    'initialdir = ' + quote(context),
    'universe = vanilla',
    'executable = build-and-test.sh',
    '+IsBuildJob = true',
    'request_cpus = 4',
    'request_memory = 16GB',
    'request_disk = 50GB',
    'should_transfer_files = YES',
    'when_to_transfer_output = ON_EXIT',
    'transfer_input_files = ' + ','.join(quote(p) for p in inputs),
    'transfer_output_files = ""',
    'log = events.log',
    'output = stdout.txt',
    'error = stderr.txt',
    'queue 1', '',
]))
print('Prepared:', context)
print('Launch from a regular CHTC terminal:')
print('  cd ' + str(context))
print('  condor_submit -i build-test.sub')
print('Then on the worker: bash build-and-test.sh')
PY
