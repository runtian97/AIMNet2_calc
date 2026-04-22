"""
Single-entry batch runner for AIMNet2 + pysisyphus.

Accepts one YAML file that points directly at multi-frame XYZ inputs.  All
reactions in the XYZ file(s) are run in parallel (one worker subprocess each)
and share a single GPU through the BatchGPUServer.

Two modes are auto-detected from the input YAML:

  • full pipeline   – preopt → NEB → TSopt → IRC → endopt
                      input:  r_xyz + p_xyz      (multi-frame, paired by index)

  • TS-only         – TSopt  → IRC → endopt
                      input:  ts_xyz             (multi-frame, one TS guess each)

Example YAML (full pipeline)
----------------------------
    reactions:
      r_xyz: /path/to/r.xyz
      p_xyz: /path/to/p.xyz

    calc:
      type:  aimnet
      model: /path/to/aimnet2nse_0.jpt
      charge: -1
      mult:   1

    workflow:
      preopt:  {max_cycles: 50}
      interpol: {type: redund, between: 10}
      cos:     {type: neb}
      opt:     {type: lbfgs, align: True, rms_force: 0.01, max_step: 0.04}
      tsopt:   {type: rsirfo, do_hess: True, max_cycles: 100,
                thresh: gau_tight, hessian_recalc: 7}
      irc:     {type: eulerpc, rms_grad_thresh: 0.0005}
      endopt:  {fragments: True}

    batch:
      workdir:  runs/         # per-reaction subdirs created here
      label_prefix: rxn       # subdirs named rxn_0000, rxn_0001, ...
      max_workers: 50         # max concurrent pysisyphus worker subprocesses
                              # (each ~500 MB RAM).  default = run all at once;
                              # set a finite cap when you have many reactions
      slim: true              # after all runs finish, keep only products
                              # up to the last successful pipeline stage

Example YAML (TS-only)
----------------------
    reactions:
      ts_xyz: /path/to/ts_guesses.xyz
    calc: ...
    workflow:
      tsopt:  {type: rsirfo, do_hess: True, ...}
      irc:    {type: eulerpc, rms_grad_thresh: 0.0005}
      endopt: {fragments: True}
    batch: {workdir: ts_runs/}

Usage
-----
    python -m aimnet2calc.batch_run input.yml
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import yaml
from pathlib import Path
from typing import List, Tuple


# ── XYZ helpers ───────────────────────────────────────────────────────────────

def parse_multi_xyz(path: Path) -> List[Tuple[int, str, list]]:
    """Return [(natoms, comment, [(sym,x,y,z), …]), …] for every frame."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        atoms = []
        for j in range(i + 2, i + 2 + n):
            cols = lines[j].split()
            atoms.append((cols[0], cols[1], cols[2], cols[3]))
        frames.append((n, comment, atoms))
        i += 2 + n
    return frames


def write_xyz_block(natoms: int, comment: str, atoms: list) -> str:
    out = [str(natoms), comment]
    for sym, x, y, z in atoms:
        out.append(f"{sym:4s} {float(x):18.10f} {float(y):18.10f} {float(z):18.10f}")
    return "\n".join(out) + "\n"


# ── per-reaction setup ────────────────────────────────────────────────────────

def build_runs(cfg: dict) -> List[Path]:
    """Materialise one workdir per reaction.  Returns the list of run dirs."""
    rcfg     = cfg["reactions"]
    bcfg     = cfg.get("batch", {})
    workflow = cfg["workflow"]
    calc     = cfg["calc"]

    workdir      = Path(bcfg.get("workdir", "runs")).resolve()
    label_prefix = bcfg.get("label_prefix", "rxn")
    workdir.mkdir(parents=True, exist_ok=True)

    if "ts_xyz" in rcfg:
        # TS-only mode: one frame per reaction
        ts_frames = parse_multi_xyz(rcfg["ts_xyz"])
        n_rxn     = len(ts_frames)
        per_rxn_geom_fn = "ts_guess.xyz"
        mode = "ts_only"
    elif "r_xyz" in rcfg and "p_xyz" in rcfg:
        r_frames = parse_multi_xyz(rcfg["r_xyz"])
        p_frames = parse_multi_xyz(rcfg["p_xyz"])
        if len(r_frames) != len(p_frames):
            raise ValueError(f"r_xyz / p_xyz frame count mismatch: "
                             f"{len(r_frames)} vs {len(p_frames)}")
        n_rxn     = len(r_frames)
        per_rxn_geom_fn = "reaction.trj"
        mode = "full"
    else:
        raise ValueError("reactions must specify either ts_xyz or "
                         "(r_xyz AND p_xyz)")

    print(f"[batch_run] {n_rxn} reactions in {mode} mode → {workdir}")

    run_dirs: List[Path] = []
    for i in range(n_rxn):
        run_dir = workdir / f"{label_prefix}_{i:04d}"
        run_dir.mkdir(exist_ok=True)

        # Per-reaction geometry file
        if mode == "ts_only":
            n, c, atoms = ts_frames[i]
            (run_dir / per_rxn_geom_fn).write_text(write_xyz_block(n, c, atoms))
        else:
            nr, cr, ar = r_frames[i]
            np_, cp, ap = p_frames[i]
            (run_dir / per_rxn_geom_fn).write_text(
                write_xyz_block(nr, cr or f"reaction {i} reactant", ar)
                + write_xyz_block(np_, cp or f"reaction {i} product", ap)
            )

        # Per-reaction pysisyphus YAML
        run_dict = dict(workflow)
        run_dict["calc"] = dict(calc)
        run_dict["geom"] = {"type": "cart", "fn": per_rxn_geom_fn}
        with open(run_dir / "input.yml", "w") as f:
            yaml.safe_dump(run_dict, f, sort_keys=False)

        run_dirs.append(run_dir)
    return run_dirs


# ── worker entrypoint ─────────────────────────────────────────────────────────

def _worker(run_dir_str: str, request_queue, response_queue, worker_id: int,
            run_dict: dict):
    """Run a single reaction in this subprocess."""
    import logging
    import traceback
    os.chdir(run_dir_str)

    # Route all calculator calls to the BatchGPUServer
    from aimnet2calc.aimnet2pysis import (
        AIMNet2Pysis, _patch_cos_for_aimnet_batch,
    )

    # pysisyphus/__init__.py attaches a FileHandler to the "pysisyphus" logger
    # with a relative path; os.path.abspath() is resolved at handler __init__
    # time which can land us in an ancestor cwd.  Replace it so each worker
    # writes to its own rxn-dir/pysisyphus.log.
    pysis_logger = logging.getLogger("pysisyphus")
    for h in list(pysis_logger.handlers):
        if isinstance(h, logging.FileHandler):
            pysis_logger.removeHandler(h)
            h.close()
    fh = logging.FileHandler(
        str(Path(run_dir_str) / "pysisyphus.log"), mode="w", delay=True,
    )
    fh.setLevel(logging.DEBUG)
    pysis_logger.addHandler(fh)

    AIMNet2Pysis.enable_remote(request_queue, response_queue, worker_id)

    from pysisyphus import run as pysis_run
    pysis_run.CALC_DICT['aimnet'] = AIMNet2Pysis
    _patch_cos_for_aimnet_batch()

    t0 = time.time()
    try:
        pysis_run.run_from_dict(run_dict, cwd=Path(run_dir_str))
        dt = time.time() - t0
        print(f"[worker {worker_id}] DONE  {Path(run_dir_str).name}  ({dt:.1f} s)",
              flush=True)
    except SystemExit:
        dt = time.time() - t0
        print(f"[worker {worker_id}] EXIT  {Path(run_dir_str).name}  ({dt:.1f} s)",
              flush=True)
    except Exception as e:
        dt = time.time() - t0
        print(f"[worker {worker_id}] FAIL  {Path(run_dir_str).name}  "
              f"({dt:.1f} s): {e}", flush=True)
        traceback.print_exc()


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_batch(cfg: dict):
    from aimnet2calc.aimnet2pysis import BatchGPUServer

    run_dirs = build_runs(cfg)

    bcfg     = cfg.get("batch", {})
    flush_s  = float(bcfg.get("flush_ms", 10)) / 1000.0
    model    = cfg["calc"]["model"]

    server = BatchGPUServer(model=model, flush_timeout_s=flush_s)

    queues, run_dicts = [], []
    for wid, rd in enumerate(run_dirs):
        with open(rd / "input.yml") as f:
            run_dicts.append(yaml.safe_load(f))
        queues.append(server.register_worker(wid))

    server.start()

    ctx = mp.get_context("spawn")

    # max_workers caps simultaneous worker subprocesses.  Each worker is a
    # full pysisyphus process (~500 MB RAM), so launching N workers for
    # N=1000 reactions will OOM most nodes.  Default to all reactions at
    # once; user can set batch.max_workers to throttle.
    max_workers = int(bcfg.get("max_workers", 0) or 0)
    if max_workers <= 0 or max_workers > len(run_dirs):
        max_workers = len(run_dirs)
    print(f"[batch_run] launching with max_workers={max_workers} "
          f"(of {len(run_dirs)} reactions)", flush=True)

    jobs = list(zip(enumerate(run_dirs), queues, run_dicts))
    t0 = time.time()
    in_flight = []   # list of currently running Process objects
    next_idx = 0
    n_done = 0
    while next_idx < len(jobs) or in_flight:
        # Fill up to max_workers
        while next_idx < len(jobs) and len(in_flight) < max_workers:
            (wid, rd), (req_q, resp_q), rd_dict = jobs[next_idx]
            p = ctx.Process(
                target=_worker,
                args=(str(rd), req_q, resp_q, wid, rd_dict),
                name=f"rxn-{rd.name}",
            )
            p.start()
            in_flight.append(p)
            next_idx += 1
        # Wait for any to finish; rescan list
        alive = []
        for p in in_flight:
            if p.is_alive():
                alive.append(p)
            else:
                p.join()
                n_done += 1
        in_flight = alive
        if next_idx < len(jobs) and len(in_flight) >= max_workers:
            # Sleep briefly to avoid busy-loop while pool is full
            time.sleep(0.5)

    print(f"\n[batch_run] {len(jobs)} reactions finished in "
          f"{time.time() - t0:.1f} s", flush=True)
    server.stop()

    if bcfg.get("slim", False):
        print("\n[batch_run] slim enabled → keeping products up to last "
              "successful stage for every run")
        from aimnet2calc.finalize import finalize_workdir
        workdir = Path(bcfg.get("workdir", "runs")).resolve()
        finalize_workdir(workdir, slim=True, verbose=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="aimnet2-batch",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("input_yaml", type=Path)
    args = ap.parse_args(argv)

    with open(args.input_yaml) as f:
        cfg = yaml.safe_load(f)

    # Resolve relative XYZ paths against the YAML's directory
    yaml_dir = args.input_yaml.resolve().parent
    for k in ("r_xyz", "p_xyz", "ts_xyz"):
        v = cfg.get("reactions", {}).get(k)
        if v is not None and not Path(v).is_absolute():
            cfg["reactions"][k] = str(yaml_dir / v)

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    run_batch(cfg)


if __name__ == "__main__":
    main()
