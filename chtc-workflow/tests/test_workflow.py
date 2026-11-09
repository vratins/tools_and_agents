import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'bin' / 'chtc'
loader = importlib.machinery.SourceFileLoader('workflow', str(CLI))
spec = importlib.util.spec_from_loader(loader.name, loader)
workflow = importlib.util.module_from_spec(spec)
loader.exec_module(workflow)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        (self.project / 'test.py').write_text('print("hello")\n')

    def prepare(self, *extra):
        subprocess.run(['python3', str(CLI), 'submit', '--project', str(self.project),
                        '--image', 'python:3.11', '--prepare', *extra, 'test.py',
                        '--literal', 'a b', '$(do_not_expand)', "quote'\""], check=True,
                       capture_output=True, text=True)
        return next((self.project / '.chtc' / 'runs').iterdir())

    def test_snapshot_and_literal_arguments(self):
        (self.project / '.env').write_text('SECRET')
        (self.project / 'data').mkdir()
        (self.project / 'data' / 'huge').write_text('not source')
        (self.project / 'outside').symlink_to('/etc/passwd')
        run = self.prepare('--gpus', '2', '--gpu-memory-mb', '10000')
        with tarfile.open(run / 'source.tar.gz') as archive:
            self.assertEqual(archive.getnames(), ['test.py'])
        manifest = json.loads((run / 'run.json').read_text())
        self.assertEqual(manifest['args'], ['--literal', 'a b', '$(do_not_expand)', "quote'\""])
        submit = (run / 'job.sub').read_text()
        self.assertIn('request_gpus = 2', submit)
        self.assertIn('gpus_minimum_memory = 10000', submit)

    def test_worker_executes_arguments_and_preserves_failure(self):
        (self.project / 'test.py').write_text('''import json, os, pathlib, sys
pathlib.Path(os.environ['CHTC_OUTPUT_DIR'], 'args.json').write_text(json.dumps(sys.argv[1:]))
assert os.environ['UV_CACHE_DIR'].startswith(str(pathlib.Path.cwd().parent))
sys.exit(7)
''')
        run = self.prepare()
        env = {**os.environ, '_CONDOR_SCRATCH_DIR': str(run)}
        result = subprocess.run(['bash', 'worker.sh'], cwd=run, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual((run / 'outputs' / 'exit-code.txt').read_text(), '7\n')
        self.assertEqual(json.loads((run / 'outputs' / 'args.json').read_text()),
                         json.loads((run / 'run.json').read_text())['args'])

    def test_invalid_resources(self):
        result = subprocess.run(['python3', str(CLI), 'interactive', '--image', 'python:3.11',
                                 '--gpus', '-1', '--prepare'], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_snapshot_size_limit(self):
        (self.project / 'large.bin').write_bytes(b'x' * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            workflow.snapshot(self.project, self.project / 'out.tgz', [], 1)

    def test_build_from_venv_generates_both_recipes(self):
        def fake_run(argv, **kwargs):
            self.assertIn('freeze', argv)
            return subprocess.CompletedProcess(argv, 0, 'torch==2.5.1+cu124\n')
        args = type('Args', (), dict(project=str(self.project), venv='.venv',
                    base='python:3.11-slim-bookworm', uv_version='0.12.23',
                    exclude=[], max_source_mb=100, extra=[], tag='test:v1', execute=False,
                    backend='apptainer', output=None, prepare=True,
                    cpus=4, memory='16GB', disk='40GB', gpus=0, gpu_memory_mb=None,
                    index_url='https://download.pytorch.org/whl/cu124'))()
        with patch.object(workflow.subprocess, 'run', side_effect=fake_run):
            workflow.build(args)
        context = next((self.project / '.chtc' / 'builds').iterdir())
        self.assertIn('/opt/venv', (context / 'Dockerfile').read_text())
        self.assertIn('%files from uv', (context / 'image.def').read_text())
        self.assertIn('+IsBuildJob = true', (context / 'build.sub').read_text())
        self.assertIn('transfer_output_files = image.sif,build-status.json',
                      (context / 'build.sub').read_text())
        self.assertFalse((context / 'source.tar.gz').exists())
        subprocess.run(['bash', '-n', str(context / 'build.sh')], check=True)

    def build_context(self):
        context = self.project / 'context'
        context.mkdir()
        state = dict(output=str(self.project / 'images' / 'env.sif'),
                     partial=str(self.project / 'images' / '.env.partial'))
        (context / 'build-state.json').write_text(json.dumps(state))
        return context, state

    def test_build_submits_waits_and_publishes(self):
        context, state = self.build_context()
        def fake_run(argv, **kwargs):
            if argv[0] == 'condor_submit':
                return subprocess.CompletedProcess(argv, 0, '123.0 - 123.0\n')
            self.assertEqual(argv[0], 'condor_wait')
            Path(state['partial']).write_bytes(b'mock-successful-image')
            (context / 'build-status.json').write_text('{"exit_code":0}')
            return subprocess.CompletedProcess(argv, 0, '')
        with patch.object(workflow.subprocess, 'run', side_effect=fake_run):
            workflow.finish_build(context)
        self.assertEqual(Path(state['output']).read_bytes(), b'mock-successful-image')
        self.assertFalse(Path(state['partial']).exists())
        self.assertEqual(json.loads((context / 'build-state.json').read_text())['job_id'], '123.0')

    def test_build_failure_never_publishes_image(self):
        context, state = self.build_context()
        Path(state['partial']).parent.mkdir()
        Path(state['partial']).write_bytes(b'broken-image')
        (context / 'build-status.json').write_text('{"exit_code":1}')
        state['job_id'] = '123.0'
        (context / 'build-state.json').write_text(json.dumps(state))
        with patch.object(workflow.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'build failed'):
                workflow.finish_build(context)
            run.assert_not_called()
        self.assertFalse(Path(state['output']).exists())
        self.assertFalse(Path(state['partial']).exists())

    def test_build_refuses_overwrite(self):
        context, state = self.build_context()
        output = Path(state['output'])
        output.parent.mkdir()
        output.write_bytes(b'existing')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            workflow.finish_build(context)
        self.assertEqual(output.read_bytes(), b'existing')

    def test_build_hold_is_reported_and_can_resume(self):
        context, state = self.build_context()
        state['job_id'] = '123.0'
        (context / 'build-state.json').write_text(json.dumps(state))
        def fake_run(argv, **kwargs):
            if argv[0] == 'condor_wait':
                return subprocess.CompletedProcess(argv, 1, '')
            self.assertEqual(argv[:2], ['condor_q', '123.0'])
            return subprocess.CompletedProcess(argv, 0, '[{"JobStatus":5,"HoldReason":"Disk limit"}]')
        with patch.object(workflow.subprocess, 'run', side_effect=fake_run):
            with self.assertRaisesRegex(ValueError, 'Disk limit'):
                workflow.finish_build(context)
        self.assertFalse(Path(state['output']).exists())

    def test_build_worker_failure_returns_status(self):
        # Exercise the real shell trap, without a container build or allocation.
        self.test_build_from_venv_generates_both_recipes()
        context = next((self.project / '.chtc' / 'builds').iterdir())
        fakebin = self.project / 'fakebin'
        fakebin.mkdir()
        apptainer = fakebin / 'apptainer'
        apptainer.write_text('#!/bin/bash\nexit 17\n')
        apptainer.chmod(0o700)
        env = {**os.environ, '_CONDOR_SCRATCH_DIR': str(context),
               'PATH': str(fakebin) + ':' + os.environ['PATH']}
        result = subprocess.run(['bash', str(context / 'build.sh')], env=env, capture_output=True)
        self.assertEqual(result.returncode, 17)
        self.assertEqual(json.loads((context / 'build-status.json').read_text()), {'exit_code': 17})
        self.assertEqual((context / 'image.sif').stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
