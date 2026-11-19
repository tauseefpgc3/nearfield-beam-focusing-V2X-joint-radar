"""D-036 Step 0, decision 1: LiDAR / power pairing per CSV row, without GPS and without shifting.
Per scenario: (a) file index in the LiDAR and power file names on every row (and vs the CSV index);
(b) CSV time_stamp spacing within sequences; (c) zip member timestamps (2 s resolution) of the
paired LiDAR and power files, and their spread within a sequence (are they capture times?).
-> results/phase4/step0_timing.json
"""
import json, os, re, sys, zipfile, datetime as dt
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "phase3c")); import ds_prepare as DP
ZIP = {31: os.path.join(ROOT, "data", "deepsense_raw", "scenario31_new.zip")} | {s: os.path.join(ROOT, "Scenario 32 33 34", z) for s, z in ((32, "scenario32.zip"), (33, "scenario33.zip"), (34, "scenario34_new.zip"))}
num = lambda p: int(re.findall(r"_(\d+)\.\w+$", p)[0])
q = lambda x: dict(min=float(np.min(x)), p5=float(np.percentile(x, 5)), median=float(np.median(x)), p95=float(np.percentile(x, 95)), max=float(np.max(x)))
R = {}
for s in (31, 32, 33, 34):
    df = pd.read_csv(os.path.join(DP.SC[s]["root"], DP.SC[s]["csv"])); seq = df.seq_index.to_numpy()
    li = np.array([num(p) for p in df.unit1_lidar]); pw = np.array([num(p) for p in df.unit1_pwr_60ghz]); ix = df["index"].to_numpy()
    t = np.array([(dt.datetime.strptime(x.split("-")[0], "%H:%M:%S") - dt.datetime(1900, 1, 1)).total_seconds() + float("0." + x.split("-")[1]) for x in df.time_stamp])
    dtt = np.concatenate([np.diff(t[seq == u]) for u in np.unique(seq)])
    Z = zipfile.ZipFile(ZIP[s]); info = {i.filename.split("/")[-1]: i.date_time for i in Z.infolist()}
    ts = lambda n: dt.datetime(*info[n]).timestamp()
    zl = np.array([ts(os.path.basename(p)) for p in df.unit1_lidar]); zp = np.array([ts(os.path.basename(p)) for p in df.unit1_pwr_60ghz])
    spread = [float(zl[seq == u].max() - zl[seq == u].min()) for u in np.unique(seq)]
    R[s] = dict(rows=len(df), lidar_index_eq_power_index=float(np.mean(li == pw)), lidar_index_eq_csv_index=float(np.mean(li == ix)),
                lidar_minus_power_index=q(li - pw), csv_timestamp_step_s=q(dtt),
                zip_mtime_lidar_minus_power_s=q(zl - zp), zip_mtime_lidar_spread_within_sequence_s=q(spread),
                csv_duration_within_sequence_s=q([t[seq == u].max() - t[seq == u].min() for u in np.unique(seq)]))
    print(s, json.dumps(R[s]), flush=True)
json.dump(R, open(os.path.join(ROOT, "results", "phase4", "step0_timing.json"), "w"), indent=1)
