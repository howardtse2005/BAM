"""Assign train / val / test splits to a manifest.

Splits are assigned per duplicate group, never per image, and are stratified by
dataset so every split sees every source. Assignment is hash-based, so rerunning
the command on the same manifest produces the same split.

    python -m training.prepare.split --ratios 0.8 0.1 0.1
    python -m training.prepare.split --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from training.prepare.dedupe import group_duplicates, stable_bucket  # noqa: E402

SPLITS = ("train", "val", "test_a")


def assign_splits(
    samples: list[dict],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    salt: str = "bam-v1",
) -> dict[str, str]:
    """Return ``sample_id -> split`` using duplicate groups and stratification."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    hashed = [(s["id"], int(s.get("phash", 0))) for s in samples]
    groups = group_duplicates(hashed)

    # Group representative -> dataset (all members share a source by construction)
    by_id = {s["id"]: s for s in samples}
    group_dataset: dict[str, str] = {}
    group_members: dict[str, list[str]] = defaultdict(list)
    for sid, gid in groups.items():
        group_members[gid].append(sid)
        group_dataset.setdefault(gid, by_id[sid].get("dataset", "unknown"))

    assignment: dict[str, str] = {}
    per_dataset: dict[str, list[str]] = defaultdict(list)
    for gid, dataset in group_dataset.items():
        per_dataset[dataset].append(gid)

    for dataset, gids in per_dataset.items():
        gids.sort(key=lambda g: (stable_bucket(g, salt), g))
        n = len(gids)
        n_val = max(1, round(n * ratios[1])) if n >= 10 else 0
        n_test = max(1, round(n * ratios[2])) if n >= 10 else 0
        # Tiny datasets: everything goes to train to avoid empty splits.
        if n < 10:
            n_val = n_test = 0
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train, n_val, n_test = n, 0, 0

        for i, gid in enumerate(gids):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test_a"
            for sid in group_members[gid]:
                assignment[sid] = split

    return assignment


def verify(samples: list[dict]) -> dict:
    """Check that no duplicate group straddles two splits."""
    hashed = [(s["id"], int(s.get("phash", 0))) for s in samples]
    groups = group_duplicates(hashed)
    by_id = {s["id"]: s for s in samples}

    members: dict[str, list[str]] = defaultdict(list)
    for sid, gid in groups.items():
        members[gid].append(sid)

    leaks = 0
    for gid, sids in members.items():
        if len({by_id[m].get("split") for m in sids}) > 1:
            leaks += 1

    counts = Counter(s.get("split") for s in samples)
    by_dataset: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        by_dataset[s.get("dataset", "?")][s.get("split")] += 1

    return {
        "samples": len(samples),
        "duplicate_groups": len(set(groups.values())),
        "group_split_leaks": leaks,
        "counts": dict(counts),
        "by_dataset": {k: dict(v) for k, v in by_dataset.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign manifest splits")
    parser.add_argument("--manifest", default="datasets/manifest.json")
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    parser.add_argument("--salt", default="bam-v1")
    parser.add_argument("--verify", action="store_true", help="only report, do not write")
    args = parser.parse_args(argv)

    path = Path(args.manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]

    if not args.verify:
        assignment = assign_splits(samples, tuple(args.ratios), args.salt)
        for sample in samples:
            sample["split"] = assignment.get(sample["id"], "train")
        data["splits"] = {
            "ratios": list(args.ratios),
            "salt": args.salt,
            "assigned": True,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    report = verify(samples)
    print(json.dumps(report, indent=2))
    if report["group_split_leaks"]:
        print("WARNING: duplicate groups cross split boundaries", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
