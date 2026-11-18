"""D-033 C3: check that every number in the paper assets traces to a banked JSON.

    python scripts/paper_v2/trace_check.py      # exit 1 on any failure; writes paper_assets_v2/trace_report.json

1. Every manifest value entry is recomputed from its JSON source (path, op, scale, format) and
   must reproduce the written text exactly; every key entry's JSON path must exist.
2. Every numeric token in canonical_macros_v2.tex and tables/*.tex (after removing LaTeX
   structure such as column counts and rule ranges) must be a registered text for that file,
   or a protocol constant (CI level, thresholds), which are listed separately.
3. Every numeric token in each figure's PDF text (pdftotext) must be a registered text for that
   figure, one of its axis tick labels, or a protocol constant.
"""
import glob, json, os, re, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper_assets_v2")
NUM = re.compile(r"(?<![A-Za-z\d.])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def norm(s): return re.sub(r"(?<=\d),(?=\d{3})", "", s.replace("−", "-").replace("{,}", "").replace("–", " "))


def tokens(s): return NUM.findall(norm(s))


def strip_tex(s):
    s = re.sub(r"(?m)^%.*$", "", s); s = s.replace("--", " ")        # comment lines; en-dash ranges
    s = s.replace("DeepSense 6G", "DeepSense")                         # dataset name, not a number
    s = re.sub(r"\\begin\{tabular\*?\}\{[^}]*\}", "", s); s = re.sub(r"\\multicolumn\{\d+\}", "", s)
    s = re.sub(r"\\cmidrule\([a-z]*\)\{\d+-\d+\}", "", s); s = re.sub(r"p\{[\d.]+cm\}", "", s)
    s = re.sub(r"\\newcommand\{\\[A-Za-z]+\}", "", s); s = re.sub(r"\\label\{[^}]*\}", "", s)
    return s


def main():
    M = json.load(open(os.path.join(OUT, "trace_manifest.json"))); const = set(M["constants"]); cache = {}; fails = []
    def get(src, path):
        if src not in cache: cache[src] = json.load(open(os.path.join(ROOT, src)))
        v = cache[src]
        for p in path: v = v[p]
        return v
    n_val = n_key = 0; reg = {}
    for it in M["items"]:
        reg.setdefault(it["target"], set()).update(tokens(it["text"]))
        if it["kind"] == "value":
            x = get(it["src"], it["path"])
            if it["op"] == "half_diff": x = (x - get(it["src"], it["path2"])) / 2
            s = it["fmt"].format(x * it["scale"])
            if s.startswith("-") and float(s) == 0: s = s[1:]
            n_val += 1
            if s != it["text"]: fails.append(f"value mismatch {it['target']} {it['src']} {it['path']}: {s} != {it['text']}")
        elif it["kind"] == "key":
            n_key += 1
            try: get(it["src"], it["path"])
            except (KeyError, IndexError, TypeError): fails.append(f"missing key {it['src']} {it['path']}")
    rep = dict(values_rechecked=n_val, keys_rechecked=n_key, files={})
    files = [os.path.join(OUT, "canonical_macros_v2.tex")] + sorted(glob.glob(os.path.join(OUT, "tables", "*.tex")))
    for f in files:
        name = os.path.basename(f); toks = tokens(strip_tex(open(f, encoding="utf8").read())); ok = reg.get(name, set())
        bad = [t for t in toks if t not in ok and t not in const]; used_const = sorted({t for t in toks if t not in ok and t in const})
        rep["files"][name] = dict(numbers=len(toks), untraced=bad, protocol_constants=used_const)
        fails += [f"{name}: untraced number {t}" for t in bad]
    for f in sorted(glob.glob(os.path.join(OUT, "figures", "*.pdf"))):
        name = os.path.basename(f)
        txt = subprocess.run(["pdftotext", "-q", "-enc", "UTF-8", f, "-"], capture_output=True).stdout.decode("utf8", "replace")
        toks = tokens(txt); ok = reg.get(name, set())
        bad = [t for t in toks if t not in ok and t not in const]; used_const = sorted({t for t in toks if t not in ok and t in const})
        rep["files"][name] = dict(numbers=len(toks), untraced=bad, protocol_constants=used_const)
        fails += [f"{name}: untraced number {t}" for t in bad]
    rep["skipped_assets"] = M["skipped"]; rep["failures"] = fails; rep["verdict"] = "PASS" if not fails else "FAIL"
    json.dump(rep, open(os.path.join(OUT, "trace_report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "files"}, indent=1))
    for k, v in rep["files"].items(): print(f"  {k}: {v['numbers']} numbers, untraced {len(v['untraced'])}, constants {v['protocol_constants']}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
