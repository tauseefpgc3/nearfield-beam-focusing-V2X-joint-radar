"""Generate the synthetic multi-modal dataset from DeepMIMO O1.

Pipeline per user (docs/04 Step 6; decisions D-008/D-009/D-010):
  1. DeepMIMO O1 ray-tracing geometry -> in-FoV per-path (azimuth, range, gain)
  2. NEAR-FIELD channel  h = Σ αₗ·spherical_steering(θₗ, rₗ)   (stored in full)
  3. exact polar label   (y_θ, y_r) = compute_optimal_label(h, codebook)   (matched filter)
  4. synthetic radar (2,256,128) and LiDAR (2,150,100), seeded per sample
  5. save HDF5: channel, radar_features, lidar_features, y_theta, y_r, distance,
                trajectory_id  (+ generation metadata & seeds)

The per-sample functions are reusable; ``main()`` drives the real O1 scenarios and
**raises** if a scenario is not downloaded (no silent fallback).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import h5py
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.codebook.polar import compute_optimal_label
from src.data.deepmimo_loader import paths_for_user, reconstruct_near_field_channel
from src.data.lidar_processing import lidar_preprocessing
from src.data.radar_processing import radar_preprocessing
from src.data.synthetic_lidar import simulate_point_cloud
from src.data.synthetic_radar import simulate_radar_iq

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_synthetic_multimodal")

# --- fixed configuration (configs/base.yaml) --------------------------------
N_ELEMENTS = 16384       # 128x128 UPA (D-011 resolved)
N_SIDE = 128
B_THETA = 64
B_R = 8
SPACING = 0.5
N_RX, N_CHIRPS, N_SAMPLES = 4, 256, 256
BASE_SEED_RADAR = 20_240_601          # D-010
BASE_SEED_LIDAR = 20_240_602
SCENARIOS = {60: "o1_60", 140: "o1_140"}

# DeepMIMO O1 selection (D-011): one BS (tx_set 3 = "BS3") and the main user grid
# (rx_set 0, ~498k users), uniformly subsampled to N_CAP_USERS to fit the ~8 GB sensor
# budget (docs/04 resource table). Subsampling is a regular index stride -> the spatial
# / distance distribution is preserved.
TX_SET_ID = 3
RX_SET_ID = 0
N_CAP_USERS = 5000
# Memory trap (D-012): at 16384 elements, channels are 256 KB each. Process users in
# chunks and stream to HDF5 — never hold all channels in RAM. Reduce CHUNK_USERS (not
# precision / user count) under memory pressure.
CHUNK_USERS = 500


def load_codebook(f_ghz: int):
    path = os.path.join(_ROOT, "data", "codebooks", f"polar_codebook_{f_ghz}GHz.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"codebook {path} missing; run scripts/generate_codebook.py first")
    data = np.load(path, allow_pickle=True)
    return data["codebook"], float(data["d_F"])


def assemble_radar_features(rd: np.ndarray, ra: np.ndarray) -> np.ndarray:
    """Stack RD (256,128) and RA (256,64->padded 128) into (2,256,128) float32 (D-009)."""
    feat = np.zeros((2, 256, 128), dtype=np.float32)
    feat[0] = rd
    feat[1, :, :ra.shape[1]] = ra
    return feat


def build_sample(idx: int, azimuths: np.ndarray, ranges: np.ndarray, gains: np.ndarray,
                 user_xy: tuple, velocity: float, distance: float, f_c: float,
                 codebook: np.ndarray) -> dict:
    """Build one sample's tensors from its per-path geometry and position."""
    h = reconstruct_near_field_channel(azimuths, ranges, gains, N=N_ELEMENTS,
                                       f_c=f_c, spacing=SPACING)
    (y_theta, y_r), _ = compute_optimal_label(h, codebook, B_THETA, B_R)

    # Radar uses the dominant (strongest-gain) path's azimuth as the look angle.
    dom = int(np.argmax(np.abs(gains)))
    iq = simulate_radar_iq(distance=distance, velocity=velocity,
                           azimuth_rad=float(azimuths[dom]), n_rx=N_RX,
                           n_chirps=N_CHIRPS, n_samples=N_SAMPLES,
                           seed=BASE_SEED_RADAR + idx)
    maps = radar_preprocessing(iq, n_rx=N_RX, n_chirps=N_CHIRPS, n_samples=N_SAMPLES)
    radar_features = assemble_radar_features(maps["RD"], maps["RA"])

    pc = simulate_point_cloud(user_xy=user_xy, user_z=0.0, seed=BASE_SEED_LIDAR + idx)
    lidar_features = lidar_preprocessing(pc)

    return {
        "channel": h.astype(np.complex128),
        "radar_features": radar_features,
        "lidar_features": lidar_features,
        "y_theta": np.int32(y_theta),
        "y_r": np.int32(y_r),
        "distance": np.float32(distance),
    }


class StreamingH5Writer:
    """Append samples to an HDF5 file in chunks (D-012: never hold all channels).

    Datasets are created resizable on the first chunk (shapes known from the data) and
    extended per chunk, so peak RAM is bounded by one chunk, not the whole dataset.
    """

    _SPECS = {
        "channel": np.complex128,
        "radar_features": np.float32,
        "lidar_features": np.float32,
        "y_theta": np.int32,
        "y_r": np.int32,
        "distance": np.float32,
        "trajectory_id": np.int32,
    }

    def __init__(self, out_path: str, f_c: float, d_f: float):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self.hf = h5py.File(out_path, "w")
        self.n = 0
        self._created = False
        self.hf.attrs.update(dict(f_c=f_c, d_F=d_f, N=N_ELEMENTS, N_side=N_SIDE,
                                  B_theta=B_THETA, B_r=B_R, spacing=SPACING,
                                  seed_radar_base=BASE_SEED_RADAR,
                                  seed_lidar_base=BASE_SEED_LIDAR))

    def _create(self, batch: dict) -> None:
        for k, dt in self._SPECS.items():
            per_shape = batch[k].shape[1:]
            self.hf.create_dataset(k, shape=(0, *per_shape), maxshape=(None, *per_shape),
                                   dtype=dt, chunks=True)
        self._created = True

    def append(self, batch: dict) -> None:
        """Append one chunk; ``batch[k]`` has leading axis = chunk size."""
        m = batch["y_theta"].shape[0]
        if m == 0:
            return
        if not self._created:
            self._create(batch)
        for k in self._SPECS:
            ds = self.hf[k]
            ds.resize(self.n + m, axis=0)
            ds[self.n:self.n + m] = batch[k]
        self.n += m

    def close(self) -> None:
        self.hf.attrs["n_samples"] = self.n
        self.hf.close()


def peak_working_set_mb() -> float:
    """Peak resident memory of this process in MB (Windows PeakWorkingSetSize)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC),
                                           wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(pmc), pmc.cb):
        raise ctypes.WinError()       # no silent fallback (RULE 2)
    return pmc.PeakWorkingSetSize / (1024 ** 2)


def select_subdataset(macro, tx_set_id: int, rx_set_id: int):
    """Pick the (tx_set, rx_set) sub-dataset from a loaded Dataset/MacroDataset.

    A filtered ``dm.load`` may return a single ``Dataset`` (one pair) or a
    ``MacroDataset`` (list of pairs); handle both.
    """
    # Single Dataset: it exposes 'txrx' directly and is not a list of sub-datasets.
    if "txrx" in getattr(macro, "keys", lambda: [])():
        t = macro["txrx"]
        if t["tx_set_id"] == tx_set_id and t["rx_set_id"] == rx_set_id:
            return macro
        raise ValueError(f"loaded Dataset is tx{t['tx_set_id']}/rx{t['rx_set_id']}, "
                         f"not tx{tx_set_id}/rx{rx_set_id}")
    for i in range(len(macro)):
        t = macro[i]["txrx"]
        if t["tx_set_id"] == tx_set_id and t["rx_set_id"] == rx_set_id:
            return macro[i]
    raise ValueError(f"no sub-dataset with tx_set_id={tx_set_id}, rx_set_id={rx_set_id}")


def generate_scenario(f_ghz: int, max_users: int = N_CAP_USERS,
                      tx_set_id: int = TX_SET_ID, rx_set_id: int = RX_SET_ID) -> str:
    """Generate the dataset for one frequency from its real DeepMIMO O1 scenario.

    The stored ``channel`` is the **near-field reconstruction** from ray-tracing
    geometry (D-008); DeepMIMO's own (planar) channel is not used, so no antenna /
    channel params are needed.
    """
    import deepmimo as dm

    scen = SCENARIOS[f_ghz]
    f_c = f_ghz * 1e9
    codebook, d_f = load_codebook(f_ghz)

    if scen not in dm.get_available_scenarios():
        raise FileNotFoundError(
            f"DeepMIMO scenario {scen!r} not available. Unzip it into "
            f"{dm.get_scenarios_dir()}. No silent fallback (RULE 2).")

    # Load ONLY the needed (tx_set, rx_set) pair, not all 54 (D-012: loading the full
    # MacroDataset peaks at ~22 GB RAM). max_paths=25 matches the system model (no silent
    # truncation at DeepMIMO's default of 10).
    macro = dm.load(scen, tx_sets=[tx_set_id], rx_sets=[rx_set_id], max_paths=25)
    d = select_subdataset(macro, tx_set_id, rx_set_id)
    tx_pos = np.asarray(d["tx_pos"]).reshape(-1)[:3]
    rx_pos = np.asarray(d["rx_pos"])
    n_total = rx_pos.shape[0]

    stride = max(1, n_total // max_users)
    user_idxs = list(range(0, n_total, stride))[:max_users]
    logger.info("%s BS%d rx%d: %d users -> subsample %d (stride %d)", scen, tx_set_id,
                rx_set_id, n_total, len(user_idxs), stride)

    out = os.path.join(_ROOT, "data", "processed", f"synthetic_{f_ghz}GHz_full.h5")
    writer = StreamingH5Writer(out, f_c, d_f)
    written, skipped = 0, 0
    chunk: list = []                       # accumulates samples for one chunk only

    def flush(chunk_samples: list) -> None:
        if not chunk_samples:
            return
        batch = {k: np.stack([s[k] for s in chunk_samples]) for k in
                 ("channel", "radar_features", "lidar_features")}
        batch["y_theta"] = np.array([s["y_theta"] for s in chunk_samples], np.int32)
        batch["y_r"] = np.array([s["y_r"] for s in chunk_samples], np.int32)
        batch["distance"] = np.array([s["distance"] for s in chunk_samples], np.float32)
        batch["trajectory_id"] = np.array([s["trajectory_id"] for s in chunk_samples],
                                          np.int32)
        writer.append(batch)

    for u in user_idxs:
        az, rng, gains = paths_for_user(d, u)
        if az.size == 0:
            skipped += 1
            continue
        rel = rx_pos[u, :2] - tx_pos[:2]
        distance = float(np.linalg.norm(rx_pos[u, :3] - tx_pos[:3]))
        s = build_sample(written + len(chunk), az, rng, gains,
                         user_xy=(rel[0], rel[1]), velocity=0.0, distance=distance,
                         f_c=f_c, codebook=codebook)
        s["trajectory_id"] = np.int32(round(rx_pos[u, 1]))  # y-row grouping (~1 m)
        chunk.append(s)
        if len(chunk) >= CHUNK_USERS:       # stream to disk, free the chunk
            flush(chunk)
            written += len(chunk)
            chunk = []
    flush(chunk)
    written += len(chunk)
    writer.close()
    logger.info("%s: %d samples written, %d users skipped (no in-FoV path) | "
                "peak RSS %.0f MB", scen, written, skipped, peak_working_set_mb())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freqs", type=int, nargs="+", default=[60, 140])
    ap.add_argument("--max-users", type=int, default=N_CAP_USERS)
    args = ap.parse_args()
    for f in args.freqs:
        generate_scenario(f, max_users=args.max_users)


if __name__ == "__main__":
    main()
