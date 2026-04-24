"""
Batch geometry optimisation for AIMNet2 + ASE LBFGS, with cross-molecule GPU
batching through the shared BatchGPUServer.

One YAML → N parallel ASE LBFGS workers, each handling one molecule.  Every
`get_forces()` call is forwarded (via multiprocessing.Queue) to the GPU server
in the main process, which groups compatible requests by (nat, charge, mult)
and issues one batched AIMNet2 forward pass per group.

Example YAML
------------
    molecules:
      mol_xyz: /path/to/mols.xyz         # multi-frame XYZ, one molecule per frame

    calc:
      type: aimnet
      model: /path/to/aimnet2nse_0.jpt
      charge: 0                          # applied to every molecule
      mult:   1

    optimization:
      fmax: 0.02                         # eV/Å convergence threshold
      max_steps: 500

    batch:
      workdir: opt_runs
      max_workers: null                  # null = all concurrent

Output
------
A single multi-frame XYZ  `<workdir>/optimized.xyz`  in input order.
Every frame's comment line carries:
    converged=<True|False>  energy=<eV>  max_force=<eV/Å>
Per-molecule scratch directories are removed after the merged file is
written.

Usage
-----
    aimnet2-batch-geom input.yml
    # or:
    python -m aimnet2calc.batch_geom_opt input.yml
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
import traceback
import yaml
from pathlib import Path


# ── XYZ I/O ───────────────────────────────────────────────────────────────────

def parse_multi_xyz(path: Path):
    """Return [(natoms, comment, [(sym, x, y, z), …]), …]."""
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


def write_xyz_block(natoms: int, comment: str, atoms: list, path: Path):
    lines = [str(natoms), comment]
    for sym, x, y, z in atoms:
        lines.append(f"{sym:4s} {float(x):18.10f} {float(y):18.10f} {float(z):18.10f}")
    Path(path).write_text("\n".join(lines) + "\n")


# ── per-molecule run-dir setup ────────────────────────────────────────────────

def build_runs(cfg: dict):
    mcfg = cfg["molecules"]
    bcfg = cfg.get("batch", {})
    workdir = Path(bcfg.get("workdir", "opt_runs")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    frames = parse_multi_xyz(mcfg["mol_xyz"])
    n = len(frames)
    print(f"[batch_geom_opt] {n} molecules → {workdir}")

    run_dirs = []
    for i, (nat, comment, atoms) in enumerate(frames):
        run_dir = workdir / f"mol_{i:04d}"
        run_dir.mkdir(exist_ok=True)
        write_xyz_block(nat, comment or f"molecule {i}", atoms,
                        run_dir / "original.xyz")
        run_dirs.append(run_dir)
    return run_dirs


# ── worker subprocess ─────────────────────────────────────────────────────────

def _worker(run_dir_str: str, request_queue, response_queue, worker_id: int,
            charge: float, mult: float, fmax: float, max_steps: int):
    """Run one molecule's ASE LBFGS optimisation in this subprocess."""
    import logging
    os.chdir(run_dir_str)

    from aimnet2calc.aimnet2ase import AIMNet2ASE
    from ase.io import read, write
    from ase.optimize.lbfgs import LBFGS

    # route calculator calls to the BatchGPUServer
    AIMNet2ASE.enable_remote(request_queue, response_queue, worker_id)

    atoms = read("original.xyz")
    calc = AIMNet2ASE(charge=charge, mult=mult)
    calc.charge = charge   # belt-and-suspenders: AIMNet2ASE has a reset quirk
    calc.mult = mult
    atoms.calc = calc
    calc.charge = charge
    calc.mult = mult

    log_path = Path("opt.log")
    t0 = time.time()
    try:
        opt = LBFGS(atoms, logfile=str(log_path))
        converged = opt.run(fmax=fmax, steps=max_steps)
        dt = time.time() - t0

        write("optimized.xyz", atoms, plain=True)
        import numpy as np
        forces = atoms.get_forces()
        energy = float(atoms.get_potential_energy())
        fmax_final = float(np.max(np.linalg.norm(forces, axis=1)))

        with open(log_path, "a") as f:
            f.write(
                f"\n# converged={converged}  "
                f"energy={energy:.6f} eV  "
                f"max_force={fmax_final:.6f} eV/Å  "
                f"steps={opt.nsteps}  "
                f"time={dt:.1f}s\n"
            )
        status = "DONE" if converged else "NOT_CONVERGED"
        print(f"[worker {worker_id}] {status:>13}  {Path(run_dir_str).name}  "
              f"({dt:.1f} s, {fmax_final:.3e} eV/Å)", flush=True)
    except Exception as e:
        dt = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"\n# FAILED: {type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
        print(f"[worker {worker_id}] FAIL          {Path(run_dir_str).name}  "
              f"({dt:.1f} s): {e}", flush=True)


# ── unified xyz output ────────────────────────────────────────────────────────

_SUMMARY_RE = re.compile(
    r"#\s*converged=(\S+)\s+energy=(\S+)\s+eV\s+max_force=(\S+)\s+eV",
)


def _finalize(workdir: Path, n_mols: int, verbose: bool = True):
    """Collect per-molecule optimised geometries into workdir/optimized.xyz
    in input-xyz order.  Each frame's comment line carries:
        converged=<True|False>  energy=<eV>  max_force=<eV/Å>
    Then remove the per-mol mol_NNNN/ subdirs.
    """
    lines: list[str] = []
    n_conv = 0
    rows   = []
    for i in range(n_mols):
        d = workdir / f"mol_{i:04d}"
        opt_xyz  = d / "optimized.xyz"
        orig_xyz = d / "original.xyz"
        log_path = d / "opt.log"

        # parse summary line from opt.log
        log_text = log_path.read_text() if log_path.exists() else ""
        m = _SUMMARY_RE.search(log_text)
        if m:
            conv   = m.group(1) == "True"
            energy = float(m.group(2))
            fmax_i = float(m.group(3))
        else:
            conv   = False
            energy = float("nan")
            fmax_i = float("nan")
        if conv:
            n_conv += 1

        # pick coords: optimised if written, else fall back to input
        xyz_src = opt_xyz if opt_xyz.exists() else orig_xyz
        if not xyz_src.exists():
            # worker crashed without even writing original — skip frame
            rows.append((i, "MISSING"))
            continue
        frame = xyz_src.read_text().splitlines()
        natoms = int(frame[0].strip())
        atom_lines = frame[2 : 2 + natoms]

        comment = (f"converged={conv}  "
                   f"energy={energy:.6f} eV  "
                   f"max_force={fmax_i:.6f} eV/Å")
        lines.append(str(natoms))
        lines.append(comment)
        lines.extend(atom_lines)
        rows.append((i, "converged ✓" if conv else "not_converged ✗"))

    (workdir / "optimized.xyz").write_text("\n".join(lines) + "\n")

    # clean per-mol subdirs
    for i in range(n_mols):
        d = workdir / f"mol_{i:04d}"
        if d.exists():
            shutil.rmtree(d)

    if verbose:
        print(f"\n{'idx':>6}  result")
        print("─" * 30)
        for i, s in rows:
            print(f"{i:>6}  {s}")
        print("─" * 30)
        print(f"Overall: {n_conv} / {n_mols} converged")
        print(f"wrote    {workdir / 'optimized.xyz'}")


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_batch(cfg: dict):
    from aimnet2calc.aimnet2pysis import BatchGPUServer

    run_dirs = build_runs(cfg)
    n = len(run_dirs)

    bcfg   = cfg.get("batch", {})
    ocfg   = cfg.get("optimization", {})
    charge = float(cfg["calc"].get("charge", 0))
    mult   = float(cfg["calc"].get("mult", 1))
    fmax   = float(ocfg.get("fmax", 0.02))
    max_steps = int(ocfg.get("max_steps", 500))
    model  = cfg["calc"]["model"]

    server = BatchGPUServer(model=model)
    queues = [server.register_worker(wid) for wid in range(n)]
    server.start()

    # cap on concurrent workers (each pysisyphus-independent ASE worker is
    # lighter than a pysisyphus worker, but still ~200 MB RAM)
    max_workers = int(bcfg.get("max_workers", 0) or 0)
    if max_workers <= 0 or max_workers > n:
        max_workers = n
    print(f"[batch_geom_opt] max_workers={max_workers} (of {n} molecules), "
          f"fmax={fmax} eV/Å, max_steps={max_steps}", flush=True)

    ctx  = mp.get_context("spawn")
    jobs = list(zip(range(n), run_dirs, queues))

    t0 = time.time()
    in_flight, next_idx = [], 0
    while next_idx < len(jobs) or in_flight:
        while next_idx < len(jobs) and len(in_flight) < max_workers:
            wid, rd, (req_q, resp_q) = jobs[next_idx]
            p = ctx.Process(
                target=_worker,
                args=(str(rd), req_q, resp_q, wid, charge, mult, fmax, max_steps),
                name=f"mol-{rd.name}",
            )
            p.start()
            in_flight.append(p)
            next_idx += 1
        alive = []
        for p in in_flight:
            if p.is_alive():
                alive.append(p)
            else:
                p.join()
        in_flight = alive
        if next_idx < len(jobs) and len(in_flight) >= max_workers:
            time.sleep(0.5)

    dt = time.time() - t0
    print(f"\n[batch_geom_opt] {n} molecules finished in {dt:.1f} s", flush=True)
    server.stop()

    _finalize(Path(bcfg.get("workdir", "opt_runs")).resolve(), n_mols=n)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="aimnet2-batch-geom",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("input_yaml", type=Path)
    args = ap.parse_args(argv)

    with open(args.input_yaml) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = args.input_yaml.resolve().parent
    v = cfg.get("molecules", {}).get("mol_xyz")
    if v and not Path(v).is_absolute():
        cfg["molecules"]["mol_xyz"] = str(yaml_dir / v)

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    run_batch(cfg)


if __name__ == "__main__":
    main()
