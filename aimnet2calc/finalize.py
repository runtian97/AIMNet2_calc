"""
Per-stage pass/fail evaluation and slim cleanup for pysisyphus run directories.

A single reaction run is "successful up to stage N" if every stage up to and
including N produced its expected output file(s) AND any post-hoc criteria
(e.g. exactly one imaginary frequency, IRC endpoints connect R/P) are met.
Stages are ordered:  preopt → neb → tsopt → irc → endopt.

Used by:
  • aimnet2calc.batch_run            (optional automatic cleanup via batch.slim)
  • batch_test/finalize_runs.py      (standalone CLI wrapper)
"""

from __future__ import annotations

import re
import shutil
import h5py
import numpy as np
from pathlib import Path
from typing import List, Tuple, Set, Dict, Optional


# Always-kept metadata files for every run, regardless of outcome
KEEP_META: Set[str] = {"pysisyphus.log", "RUN.yaml", "reaction.trj"}

# Per-stage "products" kept when that stage is judged successful.  Stages are
# evaluated in order; at the first failure we stop — later stages' products
# are not kept even if their files happen to exist.
STAGE_PRODUCTS: List[Tuple[str, Set[str]]] = [
    ("preopt", set()),                          # preopt intermediates not kept
    ("neb",    {"splined_hei.xyz"}),            # NEB TS guess
    ("tsopt",  {"ts_opt.xyz",                   # optimised TS geometry
                "ts_final_hessian.h5"}),        # hessian (contains vibfreqs)
    ("irc",    {"finished_irc.trj"}),           # full reaction path
    ("endopt", set()),                          # endopt fragments not kept
]

# Covalent radii (Å) and bond cutoff factor used for IRC connectivity check
COV_RADII: Dict[str, float] = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'F': 0.57, 'Cl': 1.02, 'Br': 1.20, 'I': 1.39,
    'S': 1.05, 'P': 1.07,
}
BOND_FACTOR: float = 1.3


# ── small XYZ / bonding helpers ───────────────────────────────────────────────

def read_xyz_frame(path: Path, index: int = -1):
    text = Path(path).read_text().splitlines()
    starts, i = [], 0
    while i < len(text):
        line = text[i].strip()
        if line and line.split()[0].isdigit():
            starts.append(i)
            i += int(line) + 2
        else:
            i += 1
    s = starts[index]
    n = int(text[s])
    syms, coords = [], []
    for j in range(s + 2, s + 2 + n):
        cols = text[j].split()
        syms.append(cols[0])
        coords.append([float(cols[1]), float(cols[2]), float(cols[3])])
    return syms, np.array(coords)


def _adjacency(symbols, coords):
    radii = np.array([COV_RADII.get(s, 0.7) for s in symbols])
    dist  = np.sqrt(((coords[:, None] - coords[None, :]) ** 2).sum(-1))
    cutoff = (radii[:, None] + radii[None, :]) * BOND_FACTOR
    return (dist < cutoff) & (dist > 0.1)


def _bonded_pairs(symbols, adj):
    n = len(symbols)
    return {frozenset((i, j)) for i in range(n) for j in range(i + 1, n) if adj[i, j]}


def _find_reaction_centre(sym_r, adj_r, adj_p):
    broken = _bonded_pairs(sym_r, adj_r) - _bonded_pairs(sym_r, adj_p)
    formed = _bonded_pairs(sym_r, adj_p) - _bonded_pairs(sym_r, adj_r)
    if not broken or not formed:
        return None, None, None
    c_idx = lg_idx = nu_idx = None
    for pair in broken:
        i, j = tuple(pair)
        si, sj = sym_r[i], sym_r[j]
        if si == 'C' or sj == 'C':
            c_idx  = i if si == 'C' else j
            lg_idx = j if si == 'C' else i
            break
    for pair in formed:
        i, j = tuple(pair)
        if c_idx is not None and (i == c_idx or j == c_idx):
            nu_idx = j if i == c_idx else i
            break
    return c_idx, lg_idx, nu_idx


def _classify_endpoint(adj, c_idx, lg_idx, nu_idx):
    if c_idx is None:
        return "unknown"
    has_lg = adj[c_idx, lg_idx]
    has_nu = adj[c_idx, nu_idx]
    if     has_lg and not has_nu: return "R-side"
    if not has_lg and     has_nu: return "P-side"
    if not has_lg and not has_nu: return "dissociated"
    return "TS-like"


# ── per-stage checks ──────────────────────────────────────────────────────────

def check_imag_freq(run_dir: Path) -> Tuple[Optional[int], Optional[float]]:
    """Return (n_imag, min_freq_cm1) from ts_final_hessian.h5, or (None, None)
    if the hessian file is missing."""
    h5_path = run_dir / "ts_final_hessian.h5"
    if not h5_path.exists():
        return None, None
    with h5py.File(h5_path, "r") as f:
        freqs = f["vibfreqs"][:]
    neg = freqs[freqs < 0]
    return int(len(neg)), (float(neg.min()) if len(neg) else 0.0)


def check_irc_connectivity(run_dir: Path):
    """Return (passes, end1_cls, end2_cls, note).

    Uses the two end frames of finished_irc.trj (which are the two true IRC
    endpoints: one for each side of the TS).  pysisyphus writes
    finished_irc.trj so frame[0] and frame[-1] are these two endpoints — the
    order (which is R-side vs P-side) is not fixed because it depends on the
    sign pysisyphus chose for the TS imaginary-mode eigenvector.  The PASS
    criterion is symmetric so either ordering is accepted.

    Reactant / product bonding patterns are read from reaction.trj (frame 0 /
    frame 1).
    """
    fin_trj = run_dir / "finished_irc.trj"
    rxn_trj = run_dir / "reaction.trj"
    if not fin_trj.exists():
        return False, "–", "–", "missing finished_irc.trj"
    if not rxn_trj.exists():
        return False, "–", "–", "missing reaction.trj"
    try:
        sym_r,   coords_r    = read_xyz_frame(rxn_trj, index=0)
        _,       coords_p    = read_xyz_frame(rxn_trj, index=1)
        sym_e1,  coords_end1 = read_xyz_frame(fin_trj, index=0)    # IRC end A
        _,       coords_end2 = read_xyz_frame(fin_trj, index=-1)   # IRC end B
    except (IndexError, ValueError) as e:
        return False, "–", "–", f"bad trj: {e}"

    adj_r    = _adjacency(sym_r,  coords_r)
    adj_p    = _adjacency(sym_r,  coords_p)
    adj_end1 = _adjacency(sym_e1, coords_end1)
    adj_end2 = _adjacency(sym_e1, coords_end2)

    c_idx, lg_idx, nu_idx = _find_reaction_centre(sym_r, adj_r, adj_p)
    if c_idx is None:
        return False, "–", "–", "can't find reaction centre"

    end1_cls = _classify_endpoint(adj_end1, c_idx, lg_idx, nu_idx)
    end2_cls = _classify_endpoint(adj_end2, c_idx, lg_idx, nu_idx)

    # one end must be R-side, the other must be P-side or dissociated
    # (the sign of the TS imaginary mode decides which is which — symmetric)
    p_side = {"P-side", "dissociated"}
    passes = (
        (end1_cls == "R-side" and end2_cls in p_side) or
        (end2_cls == "R-side" and end1_cls in p_side) or
        (end1_cls == "P-side" and end2_cls == "dissociated") or
        (end2_cls == "P-side" and end1_cls == "dissociated")
    )
    note = f"C{c_idx}-{sym_r[lg_idx]}{lg_idx}(LG)/C{c_idx}-{sym_r[nu_idx]}{nu_idx}(Nu)"
    return passes, end1_cls, end2_cls, note


def _stage_passed(run_dir: Path, name: str):
    if name == "preopt":
        ok = all((run_dir / f).exists() for f in ("first_pre_opt.xyz", "last_pre_opt.xyz"))
        return ok, "ok" if ok else "no pre-opt output"
    if name == "neb":
        ok = (run_dir / "splined_hei.xyz").exists()
        return ok, "ok" if ok else "no splined_hei.xyz"
    if name == "tsopt":
        if not (run_dir / "ts_opt.xyz").exists():
            return False, "no ts_opt.xyz"
        n_imag, min_freq = check_imag_freq(run_dir)
        if n_imag is None:
            return False, "no hessian"
        if n_imag != 1:
            return False, f"{n_imag} imag freqs"
        return True, f"1 imag ({min_freq:.0f} cm-1)"
    if name == "irc":
        if not (run_dir / "finished_irc.trj").exists():
            return False, "no finished_irc.trj"
        passes, end1_cls, end2_cls, _ = check_irc_connectivity(run_dir)
        if passes:
            return True, f"{end1_cls} / {end2_cls}"
        return False, f"bad connectivity ({end1_cls}/{end2_cls})"
    if name == "endopt":
        ok = ((run_dir / "forward_end_frag00_opt.xyz").exists()
              and (run_dir / "backward_end_frag00_opt.xyz").exists())
        return ok, "ok" if ok else "no endopt output"
    return False, f"unknown stage {name}"


# ── public API ────────────────────────────────────────────────────────────────

def evaluate_run(run_dir: Path):
    """Walk the pipeline stages in order.  Returns:
        stage_results : list of (stage_name, passed: bool, note: str)
                        truncated at the first failing stage
        keep_set      : KEEP_META ∪ products of every passing stage
        overall_pass  : True iff every stage passed
        last_ok       : name of the last stage that passed (or None)
        fail_stage    : (name, note) of the stage that failed (or None)
    """
    results, keep = [], set(KEEP_META)
    last_ok = None
    fail_stage = None
    for name, products in STAGE_PRODUCTS:
        passed, note = _stage_passed(run_dir, name)
        results.append((name, passed, note))
        if passed:
            last_ok = name
            keep |= products
        else:
            fail_stage = (name, note)
            break
    return results, keep, (fail_stage is None), last_ok, fail_stage


def slim_run(run_dir: Path, keep: Set[str]) -> int:
    """Delete everything in run_dir except files in keep. Returns count removed."""
    removed = 0
    for p in sorted(run_dir.iterdir()):
        if p.is_dir():
            shutil.rmtree(p); removed += 1
        elif p.name not in keep:
            p.unlink(); removed += 1
    return removed


def finalize_workdir(workdir: Path, slim: bool = False, verbose: bool = True,
                     split: bool = True):
    """Run evaluate_run on every rxn_*/ or run_*/ subdirectory, optionally
    slimming each to its keep-set.  When ``split=True`` (default), after
    slim every run is moved into ``<workdir>/success/<name>/`` or
    ``<workdir>/fail/<name>/`` based on its overall pass/fail.  Returns a
    list of per-run dicts."""
    workdir = Path(workdir)
    run_dirs = sorted(
        d for d in workdir.iterdir()
        if d.is_dir()
        and (d.name.startswith("rxn_") or d.name.startswith("run_"))
        and (d / "reaction.trj").exists()
    )

    rows = []
    n_pass = 0
    for run_dir in run_dirs:
        results, keep, overall, last_ok, fail_stage = evaluate_run(run_dir)
        if overall:
            n_pass += 1
        if slim:
            slim_run(run_dir, keep)
        rows.append(dict(
            label=run_dir.name,
            passed=overall,
            last_ok=last_ok,
            fail_stage=fail_stage,
            keep=sorted(keep),
            results=results,
            run_dir=run_dir,
        ))

    # ── split into success / fail subfolders ─────────────────────────────────
    if split and rows:
        success_root = workdir / "success"
        fail_root    = workdir / "fail"
        success_root.mkdir(exist_ok=True)
        fail_root.mkdir(exist_ok=True)
        for r in rows:
            src = r["run_dir"]
            # guard: already moved (e.g. re-running finalize)
            if src.parent.name in ("success", "fail"):
                continue
            if not src.exists():
                continue
            dst_root = success_root if r["passed"] else fail_root
            dst = dst_root / src.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            r["run_dir"] = dst   # update pointer

    if verbose:
        print(f"{'label':>10}  {'result':>8}  {'last OK':>10}  {'failed at':>28}")
        print("─" * 64)
        for r in rows:
            status = "PASS ✓" if r["passed"] else "FAIL ✗"
            fail_str = (f"{r['fail_stage'][0]}: {r['fail_stage'][1]}"
                        if r["fail_stage"] else "-")
            last_ok_str = r["last_ok"] or "-"
            print(f"{r['label']:>10}  {status:>8}  {last_ok_str:>10}  {fail_str:>28}")
        print("─" * 64)
        print(f"Overall: {n_pass} / {len(rows)} PASS")
        if slim:
            print(f"\n--slim: cleaned {len(rows)} run(s)")
            print(f"  Always keep : {sorted(KEEP_META)}")
            for name, prods in STAGE_PRODUCTS:
                if prods:
                    print(f"  {name:>7}: {sorted(prods)}")
        if split and rows:
            print(f"\nSplit:  {n_pass} runs → {workdir/'success'}/")
            print(f"        {len(rows) - n_pass} runs → {workdir/'fail'}/")
    return rows
