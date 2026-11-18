"""D-034 B4: acquisition time per method = M x T_beam + sensor frame latency + preprocessing +
inference, against the 263-beam sweep. Measures the new methods (1-NN search, 3-model radar
ensemble, codeword classifier) on this machine (batch 1, median of 200 after 20 warm-ups) and
reuses results/phase3b/cost.json for the rest. Run with the GPU otherwise idle.

Numerology (D-034): T_beam = one OFDM symbol + normal CP at 120 kHz SCS (8.92 us, primary);
960 kHz SCS (1.12 us) and an SSB of 4 symbols at 120 kHz (35.7 us) as sensitivity.
Sensor latency: LiDAR 10 Hz (mean age 50 ms, worst 100 ms); radar 20 Hz (25 / 50 ms).

    python scripts/phase3d/b4_timing.py   # -> results/phase3d/b4_timing.json
"""
import json, os, sys, time
import numpy as np, torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
for p in ("phase3b", "phase3", "verify3"): sys.path.insert(0, os.path.join(ROOT, "scripts", p))
sys.path.insert(0, HERE)


def tsym(scs_khz):
    mu = int(np.log2(scs_khz / 15)); return 1 / (scs_khz * 1e3) + 4.6875e-6 / 2 ** mu      # symbol + normal CP (s)


TB = dict(primary_120kHz_1sym=tsym(120), fr2_2_960kHz_1sym=tsym(960), ssb_120kHz_4sym=4 * tsym(120))
LAT = dict(lidar=dict(mean=0.050, worst=0.100), radar=dict(mean=0.025, worst=0.050))


def bench(fn, n=200, warm=20, cuda=False):
    for _ in range(warm): fn()
    ts = []
    for _ in range(n):
        if cuda: torch.cuda.synchronize()
        t = time.perf_counter(); fn()
        if cuda: torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1e3)


def main():
    cost = json.load(open(os.path.join(ROOT, "results", "phase3b", "cost.json"))); M_ = {}
    import eval_b as B
    from train_cls import Cls
    x1 = torch.randn(1, 3, 128, 208); r1 = torch.randn(1, 1, 128, 208)
    # 1-NN: pooled feature + search over the 3,005 training features (CPU)
    Ftr = F.normalize(torch.randn(3005, 32 * 52), dim=1)
    def nn_q(x): f = F.normalize(F.avg_pool2d(x[:, :1], 4).flatten(1), dim=1); return torch.cdist(f, Ftr).argmin(1)
    M_["one_nn_cpu_ms"] = bench(lambda: nn_q(x1))
    # radar ensemble: 3 radar nets in sequence
    nets = [B._load("radar", sd, "_r2")[0] for sd in (11, 22, 33)]
    with torch.no_grad():
        r1c = r1.cuda(); M_["radar_ensemble3_cuda_ms"] = bench(lambda: [n(r1c) for n in nets], cuda=True)
        cpu = [n.cpu() for n in nets]; M_["radar_ensemble3_cpu_ms"] = bench(lambda: [n(r1) for n in cpu], n=50, warm=5)
    sel = json.load(open(os.path.join(ROOT, "results", "phase3d", "cls_selection.json")))
    for f in ("60", "140"):
        h, lr = sel[f]["chosen"].split("|"); ck = torch.load(os.path.join(ROOT, "checkpoints", "phase3d", f"cls{f}_{h}_lr{float(lr):g}_s11.pt"), map_location="cpu", weights_only=False)
        m = Cls(ck["cin"], h); m.load_state_dict(ck["model"]); m.eval()
        with torch.no_grad():
            mc = m.cuda(); xc = x1.cuda(); M_[f"classifier{f}_cuda_ms"] = bench(lambda: mc(xc), cuda=True)
            m.cpu(); M_[f"classifier{f}_cpu_ms"] = bench(lambda: m(x1), n=50, warm=5)
    # per-method processing (CUDA path where available; CPU for the classical estimators)
    proc = dict(radar_picker=dict(sensor="radar", ms=cost["radar_preprocess_ms"]["cuda"] + cost["radar_picker_cpu_ms"]),
                lidar_geometric=dict(sensor="lidar", ms=cost["geometric_estimator_cpu_ms"]),
                radar_net=dict(sensor="radar", ms=cost["radar_preprocess_ms"]["cuda"] + cost["radar_net"]["inference_ms_cuda"]),
                lidar_net=dict(sensor="lidar", ms=cost["lidar_bev_ms"]["gpu_torch"] + cost["lidar_net"]["inference_ms_cuda"]),
                fused_net=dict(sensor="lidar", ms=max(cost["lidar_bev_ms"]["gpu_torch"], cost["radar_preprocess_ms"]["cuda"]) + cost["fused_net"]["inference_ms_cuda"]),
                one_nn_lidar=dict(sensor="lidar", ms=cost["lidar_bev_ms"]["gpu_torch"] + M_["one_nn_cpu_ms"]),
                one_nn_radar=dict(sensor="radar", ms=cost["radar_preprocess_ms"]["cuda"] + M_["one_nn_cpu_ms"]),
                radar_ensemble3=dict(sensor="radar", ms=cost["radar_preprocess_ms"]["cuda"] + M_["radar_ensemble3_cuda_ms"]),
                codeword_classifier=dict(sensor="lidar", ms=cost["lidar_bev_ms"]["gpu_torch"] + M_["classifier60_cuda_ms"]))
    rows = {}
    for nm, p in proc.items():
        for tb, t in TB.items():
            for M in (8, 16, 32, 64):
                air = M * t * 1e3
                rows[f"{nm}|{tb}|{M}"] = dict(air_ms=air, processing_ms=p["ms"], latency_mean_ms=air + p["ms"] + LAT[p["sensor"]]["mean"] * 1e3,
                                              latency_worst_ms=air + p["ms"] + LAT[p["sensor"]]["worst"] * 1e3)
    sweep = {tb: dict(air_ms=263 * t * 1e3) for tb, t in TB.items()}
    out = dict(T_beam_us={k: v * 1e6 for k, v in TB.items()}, sensor_latency_s=LAT, measured_ms=M_, processing_ms={k: v["ms"] for k, v in proc.items()},
               rows=rows, sweep263=sweep, hardware=cost["hardware"])
    json.dump(out, open(os.path.join(ROOT, "results", "phase3d", "b4_timing.json"), "w"), indent=1); print(json.dumps(out | dict(rows="..."), indent=1))


if __name__ == "__main__":
    main()
