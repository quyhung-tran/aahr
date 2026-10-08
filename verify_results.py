"""Compares a fresh run against the reference results shipped with the paper.

Usage:
  python verify_results.py [results_dir] [reference_dir]
      defaults: results  reference_results

Every numeric value must match within a relative tolerance of 1e-9 (the
pipeline is fully seeded, so with the library versions recorded under
"environment" in the reference files the match is exact). The "environment"
block itself is not compared. Exits non-zero on any mismatch.
"""
import json
import math
import os
import sys

FILES = ("aahr_results.json", "netsim_results.json")
REL_TOL = 1e-9


def compare(a, b, path, diffs):
    """Recursively compares two JSON values and appends one line per
    difference to diffs; path is the location used in those lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k == "environment":
                continue
            if k not in a or k not in b:
                diffs.append(f"{path}/{k}: present in only one file")
                continue
            compare(a[k], b[k], f"{path}/{k}", diffs)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: list length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", diffs)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=REL_TOL, abs_tol=1e-12):
            diffs.append(f"{path}: {a} vs {b}")
    elif a != b:
        diffs.append(f"{path}: {a!r} vs {b!r}")


def main():
    """Compares both results files and exits 1 on any difference."""
    res =sys.argv[1] if len(sys.argv) > 1 else "results"
    ref = sys.argv[2] if len(sys.argv) > 2 else "reference_results"
    failed = False
    for name in FILES:
        a = json.load(open(os.path.join(res, name)))
        b = json.load(open(os.path.join(ref, name)))
        diffs = []
        compare(a, b, name, diffs)
        if diffs:
            failed = True
            print(f"[FAIL] {name}: {len(diffs)} differences")
            for line in diffs[:20]:
                print("   ", line)
        else:
            print(f"[PASS] {name}: every value matches the reference")
        env_a, env_b = a.get("environment"), b.get("environment")
        if env_a and env_b and env_a != env_b:
            print(f"   note: library versions differ from the reference "
                  f"(yours: {env_a}; reference: {env_b})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
