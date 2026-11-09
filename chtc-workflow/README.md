# Reusable CHTC research workflow

`bin/chtc` is a standalone Python 3.9+ command with no third-party dependencies.
Use it against any project with `--project /path/to/project`. Add this toolkit's
`bin` directory to PATH, or call the executable by its full path. No files in the
existing research repositories are modified.

## 1. Develop an environment, then build an image

Prefer a uv project with `pyproject.toml` and `uv.lock`:

```bash
uv init --python 3.11
uv add torch
# In a worker allocation (or on your workstation):
uv sync
chtc build --project . --output /approved/image/location/research-v1.sif
```

An existing environment also works:

```bash
# Run package installation on a worker/workstation, not the login node.
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch
chtc build --project . --venv .venv \
  --base python:3.11-slim-bookworm \
  --output /approved/image/location/research-v1.sif
```

By default, the command exports packages, submits a CPU Apptainer build job to
CHTC's designated build workers (`+IsBuildJob = true`), and waits for completion
and image transfer. It produces a ready-to-submit `.sif` at `--output`. You do
not need Docker, a Docker Hub account, or a tag. If `--output` is omitted, it uses
a unique filename under `PROJECT/.chtc/images/`; choose an appropriate destination
explicitly for large images rather than filling home. It recreates the
environment at `/opt/venv`; it does not copy the host `.venv`. The base must contain
`python3` of the same minor version as your development environment. Override
`--base` for additional OS/CUDA libraries. GPU wheels include many CUDA userspace
libraries, but the worker provides the NVIDIA driver. Match wheel CUDA and GPU
architecture compatibility; requesting a GPU alone cannot fix an incompatible stack.

Lockfile mode excludes development groups unless needed dependencies are put in
an optional extra and selected with `--extra NAME`. Source code is not installed
in the image: each run transfers a fresh source snapshot. Packages under `src/`
are importable. Projects requiring installation-time hooks, compiled project
extensions, local/path dependencies, or a uv workspace need a customized recipe.
uv pip freeze cannot reconstruct custom index settings: provide `--index-url`
for an existing environment installed from a special wheel index. It also cannot
capture OS packages; maintain those in your base/recipe. Never put credentials in
an index URL that will be baked into an image or saved in build artifacts.

For your existing `gen_env`, run from `/home/vsrivastav28/hws_rotation`:

```bash
chtc-workflow/bin/chtc build --venv gen_env \
  --output /approved/image/location/gen-env-v1.sif
```

All build caches remain on the worker. Configure build resources with `--cpus`,
`--memory`, and `--disk` (defaults: 4, 16GB, 40GB). Recipes, logs, and submission
state are retained under `.chtc/builds/`. No source archive is made for an
environment build, so custom venv names such as `gen_env` need no exclusion.
The image is transferred to a unique partial file next to the destination, then
published only after a successful build, a Python execution check, and transfer.
Existing images are never overwritten; choose a new filename for a new version.
Held jobs and failed builds report their logs instead of claiming success.

Use `--prepare` to generate recipes without submitting or building. The printed
build context contains `Dockerfile`, `image.def`, `requirements.txt`, `build.sh`,
`build.sub`, and `build-state.json`. Submit a prepared build, or resume waiting
after interrupting the command, with:

```bash
chtc build --resume /path/to/PROJECT/.chtc/builds/image-TIMESTAMP-ID
```

Interrupting the wait leaves the build job queued/running; use its printed job ID
with `condor_rm` if you want to cancel it. Resuming does not submit a duplicate job.
Do not manually submit a prepared `build.sub`; use `--resume` so the toolkit can
track the job and finalize its transferred image.

Docker is optional. On a Docker-enabled workstation/CI:

```bash
chtc build --venv .venv --backend docker --tag YOUR_USER/research:env-v1
docker push YOUR_USER/research:env-v1
```

The Docker backend builds locally and prints the ready image tag, without
publishing. Builds target `linux/amd64`; login-node Docker builds are rejected.
`--execute` remains a legacy alias for `--backend docker`. `--output` applies
only to SIF builds. Use a new image tag for every environment change.

## 2. Submit any Python script

```bash
chtc submit --project /path/to/project \
  --image /approved/image/location/research-v1.sif \
  --gpus 1 --cpus 4 --memory 16GB --disk 40GB \
  train.py --config configs/baseline.yaml --seed 42
```

Put all workflow options **before** the script name. Everything following the
script is passed literally to Python, including spaces, quotes, and dash options;
the script's arguments are never evaluated as shell commands.

`--gpus` defaults to 0. `--gpu-memory-mb 16000` requests minimum VRAM per GPU.
Images can be registry names, `docker://...`, local `.sif` paths, or staged image
URLs, for example `osdf:///chtc/staging/v/vsrivastav28/images/env-v1.sif`.
Registry authentication is left to your HTCondor/CHTC configuration; start with
a public image. No host environment or registry credentials are forwarded.

Add `--prepare` to generate all artifacts without submitting. Otherwise the
command calls `condor_submit` immediately. Each job gets a unique private directory:

```text
PROJECT/.chtc/runs/submit-TIMESTAMP-ID/
  job.sub          # generated and editable submission
  source.tar.gz    # fixed source snapshot
  run.json         # exact image, arguments, resources, source SHA-256
  events.log       # HTCondor events, after submission
  stdout.txt
  stderr.txt
  results/         # returned outputs, run manifest, exit code
```

Write durable results into `os.environ['CHTC_OUTPUT_DIR']`. Files written elsewhere
are intentionally not returned. No automatic retry is enabled: a failed script's
exit code and output are retained for inspection. Use `condor_q`, `condor_tail JOBID`,
`condor_q -hold`, and `condor_rm JOBID` as usual. Checkpointing/resuming is the
application's responsibility; these jobs transfer results on exit, not eviction.

## 3. Interactive CPU/GPU development

```bash
chtc interactive --project /path/to/project \
  --image YOUR_DOCKERHUB_USER/research:env-v1 --gpus 1 --memory 16GB
# Once HTCondor opens the worker shell:
bash worker.sh --interactive
python train.py --config configs/small.yaml
```

That one bootstrap command extracts your snapshot, redirects caches into scratch,
and opens a shell with the right Python path. HTCondor's interactive mode opens a
shell instead of running the batch executable, so the bootstrap is explicit.
Interactive outputs are temporary: copy valuable notebooks/results to an approved
persistent destination **before exiting**. Edits to the original project do not
sync to an already-running snapshot. Exit the worker shell and the allocation when
finished. VS Code's other terminals and Run buttons still target the login node.
Shared GPU interactive sessions are currently limited to four hours; GPU count and
VRAM requests must fit an available interactive slot.

## Data, caches, and storage

The snapshot excludes `.venv`, `.git`, `.chtc`, credentials such as `.env`/private
keys, and common cache/data/output directories. Symlinks are skipped rather than
followed. This is a conservative convenience filter, not a secret scanner: review
your source before publishing/sharing it. `--exclude 'pattern'` can be repeated.
Source defaults to a 100MB uncompressed cap so datasets are not silently shipped
with code. Review old `.chtc/runs` periodically; source snapshots and results persist.

Use `--input /path/to/file` or `--input osdf:///chtc/staging/.../dataset.tar` for
data. Input basenames must be unique. Files arrive in `CHTC_INPUT_DIR`, the job
scratch root. Unpack inputs there in your script. Use the documented CHTC transfer
protocol for your file size; large files should not travel through home. Request
enough disk for the image, compressed and extracted inputs, caches, and outputs.
Cache directories for uv, Hugging Face, PyTorch, Triton, Jupyter, and temporary files
are all placed in worker scratch. Reusable model weights should be transferred as
an input bundle, not downloaded on every run. No cache directory is created in staging.

Your staging quota was 76.7/100GB and 964/1000 items when inspected on November 8,
2026. Run `get_quotas` again before storing an image. Stage large bundles, not
virtual environments or thousands of cache files. Keep backups outside CHTC.

## GPU notebooks in VS Code

The notebook's **kernel/server must run in the GPU allocation**. A Python kernel
selected on the login node will not gain access to the job's GPU. Include
`jupyterlab` and `ipykernel` in the container environment, start an interactive GPU
allocation, bootstrap it, and start Jupyter on the worker:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Keep token authentication enabled. VS Code's notebook kernel picker can connect to
an **Existing Jupyter Server** using its reachable URL and token. However, this
requires a supported tunnel to the worker's loopback port. A plain forward to the
login node's port 8888 does not reach that server, and VS Code Remote SSH on the
login node does not automatically forward a worker's loopback port.

CHTC disables `condor_ssh_to_job` on shared GPU machines and directs users to
interactive jobs. This toolkit does not promise a working SSH/port-forwarding
route for shared GPU notebooks: confirm the supported tunnel with CHTC facilitators
for your pool, or use a notebook-oriented GPU service. Researcher-owned resources
may provide an approved route. Do not expose an unauthenticated notebook on all
interfaces as a workaround. BadgerCompute provides Jupyter hosting, but verify GPU
availability for your needs rather than assuming it has the same GPUs as HTC.

References: [CHTC GPU interactive policy](https://chtc.cs.wisc.edu/uw-research-computing/htc-interactive-gpu-jobs/),
[VS Code Jupyter kernels](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management),
[CHTC containers](https://chtc.cs.wisc.edu/uw-research-computing/docker-jobs/),
[CHTC Apptainer builds](https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc/),
[uv container integration](https://docs.astral.sh/uv/guides/integration/docker/).

## Python 3.11 + PyTorch GPU smoke test

From this toolkit directory:

```bash
bash examples/gpu-smoke/prepare.sh
cd examples/gpu-smoke/.chtc/build-test
condor_submit -i build-test.sub
# On the allocated build worker:
bash build-and-test.sh
```

This creates a real Python 3.11 uv environment in worker scratch, installs
`torch==2.5.1` CUDA 12.4 wheels, generates a container from that environment, builds
the SIF, and checks Python/torch imports inside it. The pinned wheel is a small
reproducibility target, not a recommendation for every GPU: CUDA 12.4 requires a
compatible driver, and this torch release does not target all newer architectures.
The test uses 50GB worker disk and 16GB memory. No GPU is reserved during the build.
Copy the resulting image to an approved persistent image path before exiting.

Then, from the login node (adjust the image URL):

```bash
bin/chtc submit --project examples/gpu-smoke \
  --image osdf:///chtc/staging/v/vsrivastav28/images/gpu-smoke.sif \
  --gpus 1 --cpus 2 --memory 8GB --disk 20GB \
  smoke.py --expected-gpus 1
```

The test asserts Python 3.11 and CUDA visibility, checks the allocated GPU count,
runs matrix multiplication/backpropagation on each GPU, checks finite gradients,
compares a small result against the CPU, and saves `gpu-smoke.json` with GPU names,
versions, timing, and peak GPU memory. A CPU-only execution must fail. Read the
returned JSON and `exit-code.txt`; a queued job alone is not a successful test.

Local checks (no GPU or submission):

```bash
python3 -m unittest discover -s tests -v
```

These check source isolation, literal argument handling, resource validation,
container recipes, and worker execution/exit-code propagation. They do not claim
to validate GPU hardware, scheduler connectivity, or a completed container build.

Validation in the agent session on November 9, 2026: all ten local tests passed,
including mocked scheduler success/hold/failure paths, overwrite protection, and
the actual build worker shell's failure-status trap.
Actual scheduling was blocked: the session could not discover a usable network
interface for `condor_q`. A submit dry-run also failed on that interface; with an
explicit loopback interface it reached site credential processing and failed to
contact CredD. No container build, torch installation, or GPU job has run in this
session. Use a normal CHTC terminal for the prepared end-to-end smoke workflow.
