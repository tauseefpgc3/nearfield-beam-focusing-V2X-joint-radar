"""B8 (pre-registered D-028): parameter counts and batch-1 inference latency. Eval-only.

    python scripts/revision/b8_cost.py      # run with the GPU otherwise idle
"""
from __future__ import annotations

import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import dump, load_model                              # noqa: E402
from scripts.eval_fusion_rules import combine                     # noqa: E402
from src.data.radar_processing import radar_preprocessing         # noqa: E402
from src.data.synthetic_radar import simulate_radar_iq            # noqa: E402
from src.models.sensing_polar import BRANCHES                     # noqa: E402


def softmax_stack(o, kind):
    return torch.stack([o[f"{b}_{kind}_logits"].float().softmax(1) for b in BRANCHES])


def make_fns(dev):
    r = [load_model("radar", "60", s, dev) for s in (11, 22, 33)]
    fz = load_model("fusion", "60", 11, dev)

    def radar(xr, xl):
        o = r[0](xr)
        return o["theta_logits"].float().softmax(1), o["range_logits"].float().softmax(1)

    def full(xr, xl):
        o = fz(xr, xl)
        pth, prg = softmax_stack(o, "theta"), softmax_stack(o, "range")
        return combine("soft_avg", pth, prg, pth.max(-1).values, prg.max(-1).values)

    def ens3(xr, xl):
        os_ = [m(xr) for m in r]
        pth = torch.stack([o["theta_logits"].float().softmax(1) for o in os_])
        prg = torch.stack([o["range_logits"].float().softmax(1) for o in os_])
        w_t = pth.max(-1).values; w_r = prg.max(-1).values
        w_t = w_t / w_t.sum(0, keepdim=True); w_r = w_r / w_r.sum(0, keepdim=True)
        return (w_t.unsqueeze(-1) * pth).sum(0), (w_r.unsqueeze(-1) * prg).sum(0)

    params = {"radar_branch_alone": sum(p.numel() for p in r[0].parameters()),
              "full_three_branch": sum(p.numel() for p in fz.parameters()),
              "radar_3x_ensemble": 3 * sum(p.numel() for p in r[0].parameters())}
    sub = {"fusion_submodules": {n: sum(p.numel() for p in m.parameters()) for n, m in fz.named_children()}}
    return {"radar_branch_alone": radar, "full_three_branch": full, "radar_3x_ensemble": ens3}, params, sub


@torch.no_grad()
def bench(fn, xr, xl, dev, warm, iters, amp):
    ts = []
    for i in range(warm + iters):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        if amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                fn(xr, xl)
        else:
            fn(xr, xl)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            ts.append((time.perf_counter() - t) * 1e3)
    return dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)), n=iters)


def main():
    out = {"prereg": "docs/decisions.md D-028, release commit 99b5cda", "batch_size": 1,
           "excludes": "sensor preprocessing (FFT/BEV) unless stated",
           "hardware": dict(gpu=torch.cuda.get_device_name(0), cpu=platform.processor(),
                            cpu_logical=os.cpu_count(), torch=torch.__version__), "latency": {}}
    for devname in ("cuda", "cpu"):
        dev = torch.device(devname)
        fns, params, sub = make_fns(dev)
        out["params"] = params; out.update(sub)
        xr = torch.rand(1, 2, 256, 128, device=dev); xl = torch.rand(1, 2, 150, 100, device=dev)
        if devname == "cuda":
            for name, fn in fns.items():
                out["latency"][f"gpu_fp32__{name}"] = bench(fn, xr, xl, dev, 50, 300, False)
                out["latency"][f"gpu_fp16amp__{name}"] = bench(fn, xr, xl, dev, 50, 300, True)
        else:
            default_threads = torch.get_num_threads()
            out["cpu_default_threads"] = default_threads
            for th in (default_threads, 1):
                torch.set_num_threads(th)
                for name, fn in fns.items():
                    out["latency"][f"cpu_fp32_{th}thr__{name}"] = bench(fn, xr, xl, dev, 10, 100, False)
            torch.set_num_threads(default_threads)
        for k, v in out["latency"].items():
            if k.startswith("gpu" if devname == "cuda" else "cpu"):
                print(f"  {k}: median {v['median_ms']:.2f} ms p90 {v['p90_ms']:.2f} ms", flush=True)
    iq = simulate_radar_iq(10.0, 0.0, 0.2, seed=0)
    ts = []
    for i in range(23):
        t = time.perf_counter(); radar_preprocessing(iq); ts.append((time.perf_counter() - t) * 1e3)
    out["cpu_radar_preprocessing_one_cube_ms_median"] = float(np.median(ts[3:]))
    print(out["params"], out["cpu_radar_preprocessing_one_cube_ms_median"])
    dump("B8_cost.json", out)


if __name__ == "__main__":
    main()
