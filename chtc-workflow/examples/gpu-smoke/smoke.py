"""Fail unless PyTorch can compute and backpropagate on the allocated GPU(s)."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--expected-gpus', type=int, default=1)
args = parser.parse_args()
assert sys.version_info[:2] == (3, 11), sys.version
assert args.size > 0 and args.steps > 0
assert torch.cuda.is_available(), 'CUDA unavailable: this test requires an allocated GPU'
assert torch.cuda.device_count() == args.expected_gpus, (
    f'Expected {args.expected_gpus} GPUs, found {torch.cuda.device_count()}')
results = []
for index in range(torch.cuda.device_count()):
    device = torch.device(f'cuda:{index}')
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(device)
    a = torch.randn(args.size, args.size, device=device, requires_grad=True)
    b = torch.randn(args.size, args.size, device=device)
    torch.cuda.synchronize(device)
    start = time.monotonic()
    for _ in range(args.steps):
        a.grad = None
        product = a @ b
        loss = product.square().mean()
        loss.backward()
        assert loss.is_cuda and a.grad.is_cuda
        assert torch.isfinite(loss).item() and torch.isfinite(a.grad).all().item()
    # Small independently calculated CPU result checks numerical correctness.
    reference = a[:16, :16].detach().cpu() @ b[:16, :16].cpu()
    actual = a[:16, :16].detach() @ b[:16, :16]
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    torch.cuda.synchronize(device)
    results.append(dict(device=index, name=torch.cuda.get_device_name(index),
                        seconds=time.monotonic() - start, loss=loss.item(),
                        peak_memory_bytes=torch.cuda.max_memory_allocated(device)))
report = dict(status='PASS', python=sys.version, torch=torch.__version__,
              cuda=torch.version.cuda, visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
              size=args.size, steps=args.steps, devices=results)
output = Path(os.environ['CHTC_OUTPUT_DIR'])
output.mkdir(parents=True, exist_ok=True)
(output / 'gpu-smoke.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
