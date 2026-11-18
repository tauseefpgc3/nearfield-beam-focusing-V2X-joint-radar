"""Independent verification, step 5: V11 parameters and batch-1 latency.
Radar model, 3x radar ensemble (+soft-average), full SensingPolarPredictor (+soft-average).
Output results/verify/v5_cost.json"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.models.radar_encoder import RadarPolarModel
from src.models.sensing_polar import SensingPolarPredictor


def soft(ps):
    P = torch.stack(ps); w = P.max(-1).values; w = w / w.sum(0, keepdim=True)
    return (w[..., None] * P).sum(0)


def run(dev, fp16, n_warm, n_it):
    r = [RadarPolarModel().to(dev).eval() for _ in range(3)]
    fm = SensingPolarPredictor().to(dev).eval()
    xr = torch.randn(1, 2, 256, 128, device=dev); xl = torch.randn(1, 2, 150, 100, device=dev)

    def f_radar():
        o = r[0](xr); return o["theta_logits"].softmax(1), o["range_logits"].softmax(1)

    def f_ens():
        os_ = [m(xr) for m in r]
        return soft([o["theta_logits"].float().softmax(1) for o in os_]), soft([o["range_logits"].float().softmax(1) for o in os_])

    def f_fus():
        o = fm(xr, xl)
        return (soft([o[f"{b}_theta_logits"].float().softmax(1) for b in ("radar", "lidar", "fused")]),
                soft([o[f"{b}_range_logits"].float().softmax(1) for b in ("radar", "lidar", "fused")]))
    out = {}
    for name, fn in (("radar", f_radar), ("radar_x3_ensemble", f_ens), ("fusion_3branch", f_fus)):
        ts = []
        with torch.no_grad(), torch.amp.autocast(dev.type, dtype=torch.float16, enabled=fp16):
            for i in range(n_warm + n_it):
                if dev.type == "cuda": torch.cuda.synchronize()
                t = time.perf_counter(); fn()
                if dev.type == "cuda": torch.cuda.synchronize()
                if i >= n_warm: ts.append((time.perf_counter() - t) * 1e3)
        out[name] = dict(median_ms=float(np.median(ts)), p90_ms=float(np.percentile(ts, 90)))
    return out


res = dict(params=dict(radar=sum(p.numel() for p in RadarPolarModel().parameters()),
                       radar_x3=3 * sum(p.numel() for p in RadarPolarModel().parameters()),
                       fusion_3branch=sum(p.numel() for p in SensingPolarPredictor().parameters())))
res["gpu_fp32"] = run(torch.device("cuda"), False, 50, 300)
res["gpu_fp16"] = run(torch.device("cuda"), True, 50, 300)
res["cpu_fp32"] = run(torch.device("cpu"), False, 5, 40)
torch.set_num_threads(1)
res["cpu_fp32_1thread"] = run(torch.device("cpu"), False, 3, 20)
res["hw"] = dict(gpu=torch.cuda.get_device_name(0))
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(ROOT, "results", "verify", "v5_cost.json"), "w"), indent=1)
