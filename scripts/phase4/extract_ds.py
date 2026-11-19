"""D-036 Step 0: copy the needed members (CSV, unit1 LiDAR / mmWave / GPS, unit2 GPS, resources)
of the downloaded DeepSense zips into data/deepsense_raw/ex/s{S}/scenario{S}/. The zips are only
read, never modified; radar and camera members are skipped (not used by the pipeline).

    python scripts/phase4/extract_ds.py 32 33 34
"""
import os, sys, zipfile
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "Scenario 32 33 34"); ZIP = {32: "scenario32.zip", 33: "scenario33.zip", 34: "scenario34_new.zip"}
KEEP = ("unit1/lidar_data/", "unit1/mmWave_data/", "unit1/GPS_data/", "unit2/GPS_data/", "resources/")
for s in map(int, sys.argv[1:]):
    z = zipfile.ZipFile(os.path.join(SRC, ZIP[s])); dst = os.path.join(ROOT, "data", "deepsense_raw", "ex", f"s{s}", f"scenario{s}"); n = 0
    for i in z.infolist():
        if i.filename.endswith("/"): continue
        rel = i.filename[len(f"scenario{s}/"):] if i.filename.startswith(f"scenario{s}/") else i.filename
        if not (rel.endswith(".csv") and "/" not in rel) and not rel.startswith(KEEP): continue
        out = os.path.join(dst, rel); os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) == i.file_size: n += 1; continue
        with z.open(i) as a, open(out, "wb") as b:
            while True:
                buf = a.read(1 << 20)
                if not buf: break
                b.write(buf)
        n += 1
    print(s, "members copied", n, flush=True)
