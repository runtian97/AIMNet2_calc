"""
Batch single-point AIMNet2 calculation: energy / forces / hessian, optionally
plus thermal properties (enthalpy, entropy, Gibbs energy from IdealGasThermo).

One-shot computation per molecule — no iterative state, no subprocesses,
no queue.  Main process loads the model once, groups molecules by atom
count, and issues one batched AIMNet2 forward per group.  Large groups
can be split via `batch.max_batch_size` to stay within GPU memory.

Output: a single HDF5 file `<workdir>/results.h5`, indexed by atom count
(`NNN/` zero-padded subgroups).

Example YAML
------------
    molecules:
      mol_xyz: /path/to/mols.xyz

    calc:
      type: aimnet
      model: /path/to/aimnet2nse_0.jpt
      charge: 0
      mult:   1

    properties:
      energy:  true
      forces:  true
      hessian: true          # required if thermal.enabled is true

    thermal:
      enabled: true                # flip to false (or delete the block) to skip
      temperature:     298.15      # K
      pressure:        101325      # Pa
      geometry:        nonlinear   # linear / nonlinear / monatomic
      symmetry_number: 1
      ignore_imag_modes: true

    batch:
      workdir: calc_runs
      max_batch_size: null         # null = one forward per atom-count group

Usage
-----
    aimnet2-batch-calc input.yml
"""

from __future__ import annotations

import argparse
import h5py
import numpy as np
import time
import torch
import yaml
from collections import defaultdict
from pathlib import Path

from aimnet2calc import AIMNet2Calculator
from pysisyphus.elem_data import ATOMIC_NUMBERS


# ── XYZ parsing ───────────────────────────────────────────────────────────────

def parse_multi_xyz(path: Path):
    """Return [(natoms, comment, symbols, coords_ang), ...] for each frame."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].strip())
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        symbols, coords = [], []
        for j in range(i + 2, i + 2 + n):
            cols = lines[j].split()
            symbols.append(cols[0])
            coords.append([float(cols[1]), float(cols[2]), float(cols[3])])
        frames.append((n, comment, symbols, np.array(coords, dtype=np.float64)))
        i += 2 + n
    return frames


# ── core batched AIMNet2 compute ──────────────────────────────────────────────

def _batch_compute(model, symbols_list, coords_list, charge: float, mult: float,
                   do_forces: bool, do_hessian: bool):
    """One batched AIMNet2 forward pass.  Inputs in Å, outputs in eV /
    eV·Å⁻¹ / eV·Å⁻².  Bypasses AIMNet2's `mol_idx[-1] > 0` rejection when
    hessian is requested (per-molecule hessian blocks verified identical
    to single-molecule calls)."""
    B = len(symbols_list)
    nat = len(symbols_list[0])
    device = model.device

    numbers = torch.stack([
        torch.as_tensor(
            [ATOMIC_NUMBERS[s.lower()] for s in syms],
            dtype=torch.int, device=device,
        )
        for syms in symbols_list
    ])
    coords_t = torch.stack([
        torch.as_tensor(c, dtype=torch.float, device=device).view(nat, 3)
        for c in coords_list
    ])
    charge_t = torch.full((B,), charge, dtype=torch.float, device=device)
    mult_t   = torch.full((B,), mult,   dtype=torch.float, device=device)
    data_in  = dict(coord=coords_t, numbers=numbers, charge=charge_t, mult=mult_t)

    if do_hessian:
        data = model.prepare_input(data_in)
        data = model.set_grad_tensors(data, forces=True, stress=False, hessian=True)
        with torch.jit.optimized_execution(False):
            data = model.model(data)
        data = model.get_derivatives(data, forces=True, stress=False, hessian=True)
        data = model.process_output(data)
        hess_full = data['hessian']
        hess_5d   = hess_full.view(B, nat, 3, B, nat, 3)
        hess_per_mol = torch.stack([hess_5d[b, :, :, b, :, :] for b in range(B)])
    else:
        data = model(data_in, forces=do_forces)

    results = []
    for i in range(B):
        r = {'energy': float(data['energy'][i].item())}
        if do_forces or do_hessian:
            r['forces'] = data['forces'][i].detach().cpu().numpy().astype(np.float64)
        if do_hessian:
            h = hess_per_mol[i].detach().cpu().numpy()     # (N, 3, N, 3)
            r['hessian']    = h.reshape(nat * 3, nat * 3).astype(np.float64)
            r['hessian_4d'] = h.astype(np.float64)          # for ASE VibrationsData
        results.append(r)
    return results


# ── thermal properties (per molecule, CPU) ────────────────────────────────────

def _compute_thermal(symbols, coords, energy, hessian_4d, charge, mult, tcfg):
    """ASE IdealGasThermo per molecule.  Returns enthalpy (eV),
    entropy (eV/K), gibbs_energy (eV)."""
    from ase import Atoms
    from ase.vibrations import VibrationsData
    from ase.thermochemistry import IdealGasThermo

    numbers = np.array([ATOMIC_NUMBERS[s.lower()] for s in symbols], dtype=np.int32)
    atoms = Atoms(numbers=numbers, positions=coords)
    vib = VibrationsData(atoms, hessian_4d)
    vib_energies = vib.get_energies()            # used internally only

    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        potentialenergy=float(energy),
        atoms=atoms,
        geometry=tcfg.get("geometry", "nonlinear"),
        symmetrynumber=int(tcfg.get("symmetry_number", 1)),
        spin=(float(mult) - 1.0) / 2.0,
        ignore_imag_modes=bool(tcfg.get("ignore_imag_modes", True)),
    )

    T = float(tcfg.get("temperature", 298.15))
    P = float(tcfg.get("pressure",    101325.0))
    H = thermo.get_enthalpy(temperature=T, verbose=False)
    S = thermo.get_entropy(temperature=T, pressure=P, verbose=False)
    G = thermo.get_gibbs_energy(temperature=T, pressure=P, verbose=False)
    return dict(enthalpy=float(H), entropy=float(S), gibbs_energy=float(G))


# ── orchestrator ──────────────────────────────────────────────────────────────

def run_batch(cfg: dict):
    mcfg   = cfg["molecules"]
    bcfg   = cfg.get("batch", {})
    pcfg   = cfg.get("properties", {})
    tcfg   = cfg.get("thermal")              # None if absent (= thermal off)
    charge = float(cfg["calc"].get("charge", 0))
    mult   = float(cfg["calc"].get("mult", 1))
    model_path = cfg["calc"]["model"]

    do_energy  = bool(pcfg.get("energy",  True))
    do_forces  = bool(pcfg.get("forces",  True))
    do_hessian = bool(pcfg.get("hessian", True))
    # thermal is on iff the `thermal:` block exists AND `thermal.enabled`
    # is not explicitly false (default True when the block is present)
    do_thermal = bool(tcfg is not None and tcfg.get("enabled", True))
    if do_thermal and not do_hessian:
        raise ValueError("thermal.enabled is true but properties.hessian is "
                         "false — thermal properties need the hessian.")

    props = []
    if do_energy:  props.append("energy (eV)")
    if do_forces:  props.append("forces (eV/Å)")
    if do_hessian: props.append("hessian (eV/Å²)")
    if do_thermal:
        props.extend(["enthalpy (eV)", "entropy (eV/K)", "gibbs_energy (eV)"])

    workdir = Path(bcfg.get("workdir", "calc_runs")).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    max_chunk = int(bcfg.get("max_batch_size") or 0) or None

    frames = parse_multi_xyz(mcfg["mol_xyz"])
    n = len(frames)
    print(f"[batch_calc] {n} molecules from {mcfg['mol_xyz']}")
    print(f"[batch_calc] properties: {props}")
    if do_thermal:
        print(f"[batch_calc] thermal: T={tcfg.get('temperature', 298.15)} K, "
              f"P={tcfg.get('pressure', 101325.0)} Pa, "
              f"geometry={tcfg.get('geometry', 'nonlinear')}, "
              f"symmetry_number={tcfg.get('symmetry_number', 1)}")
    if max_chunk:
        print(f"[batch_calc] max_batch_size per nat group: {max_chunk}")

    # group by atom count
    groups: dict[int, list] = defaultdict(list)
    for idx, (nat, comment, symbols, coords) in enumerate(frames):
        groups[nat].append((idx, comment, symbols, coords))

    print(f"[batch_calc] loading {model_path} ...", flush=True)
    model = AIMNet2Calculator(model_path)

    per_nat = defaultdict(lambda: {
        "mol_xyz_idx": [], "coord": [], "numbers": [],
        "charge": [], "mult": [],
        "energy": [], "forces": [], "hessian": [],
        "enthalpy": [], "entropy": [], "gibbs_energy": [],
    })
    failed: list[tuple[int, str]] = []

    t0 = time.time()
    for nat in sorted(groups):
        mols = groups[nat]
        chunks = ([mols[i:i + max_chunk] for i in range(0, len(mols), max_chunk)]
                  if max_chunk else [mols])

        for chunk in chunks:
            symbols_list = [m[2] for m in chunk]
            coords_list  = [m[3] for m in chunk]
            try:
                res_list = _batch_compute(
                    model, symbols_list, coords_list, charge, mult,
                    do_forces=(do_forces or do_hessian),
                    do_hessian=do_hessian,
                )
                for (idx, _comment, symbols, coords), r in zip(chunk, res_list):
                    bucket = per_nat[nat]
                    bucket["mol_xyz_idx"].append(idx)
                    bucket["coord"].append(coords)
                    bucket["numbers"].append(
                        [ATOMIC_NUMBERS[s.lower()] for s in symbols]
                    )
                    bucket["charge"].append(charge)
                    bucket["mult"].append(mult)
                    if do_energy:
                        bucket["energy"].append(r["energy"])
                    if do_forces:
                        bucket["forces"].append(r["forces"])
                    if do_hessian:
                        bucket["hessian"].append(r["hessian"])
                    if do_thermal:
                        th = _compute_thermal(
                            symbols, coords, r["energy"], r["hessian_4d"],
                            charge, mult, tcfg,
                        )
                        bucket["enthalpy"].append(th["enthalpy"])
                        bucket["entropy"].append(th["entropy"])
                        bucket["gibbs_energy"].append(th["gibbs_energy"])
                print(f"[batch_calc] nat={nat:>3}  B={len(chunk):>3}  ok", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                msg = f"{type(e).__name__}: {e}"
                for (idx, _, _, _) in chunk:
                    failed.append((idx, msg))
                print(f"[batch_calc] nat={nat:>3}  B={len(chunk):>3}  FAILED: {e}",
                      flush=True)

    dt = time.time() - t0
    n_ok = sum(len(d["mol_xyz_idx"]) for d in per_nat.values())

    # ── write unified h5 ──────────────────────────────────────────────────────
    out_path = workdir / "results.h5"
    with h5py.File(out_path, "w") as h5:
        h5.attrs["n_molecules_input"]   = n
        h5.attrs["n_molecules_success"] = n_ok
        h5.attrs["n_molecules_failed"]  = len(failed)
        h5.attrs["input_xyz"]           = mcfg["mol_xyz"]
        h5.attrs["elapsed_s"]           = dt
        h5.attrs["properties_computed"] = np.array(
            props, dtype=h5py.string_dtype(encoding="utf-8"),
        )
        if do_thermal:
            h5.attrs["thermal_temperature_K"]     = float(tcfg.get("temperature", 298.15))
            h5.attrs["thermal_pressure_Pa"]       = float(tcfg.get("pressure",    101325.0))
            h5.attrs["thermal_geometry"]          = str(tcfg.get("geometry",      "nonlinear"))
            h5.attrs["thermal_symmetry_number"]   = int(tcfg.get("symmetry_number", 1))
            h5.attrs["thermal_ignore_imag_modes"] = bool(
                tcfg.get("ignore_imag_modes", True)
            )

        for nat, bucket in sorted(per_nat.items()):
            if not bucket["mol_xyz_idx"]:
                continue
            grp = h5.create_group(f"{nat:03d}")
            grp.create_dataset("mol_xyz_idx",
                               data=np.array(bucket["mol_xyz_idx"], dtype=np.int32))
            grp.create_dataset("coord",   data=np.stack(bucket["coord"]))
            grp.create_dataset("numbers", data=np.array(bucket["numbers"], dtype=np.int32))
            grp.create_dataset("charge",  data=np.array(bucket["charge"],  dtype=np.float64))
            grp.create_dataset("mult",    data=np.array(bucket["mult"],    dtype=np.float64))
            if do_energy:
                grp.create_dataset("energy", data=np.array(bucket["energy"], dtype=np.float64))
            if do_forces:
                grp.create_dataset("forces", data=np.stack(bucket["forces"]))
            if do_hessian:
                grp.create_dataset("hessian", data=np.stack(bucket["hessian"]))
            if do_thermal:
                grp.create_dataset("enthalpy",     data=np.array(bucket["enthalpy"],     dtype=np.float64))
                grp.create_dataset("entropy",      data=np.array(bucket["entropy"],      dtype=np.float64))
                grp.create_dataset("gibbs_energy", data=np.array(bucket["gibbs_energy"], dtype=np.float64))

        if failed:
            fgrp = h5.create_group("failed")
            fgrp.create_dataset("mol_xyz_idx",
                                data=np.array([f[0] for f in failed], dtype=np.int32))
            fgrp.create_dataset("messages",
                                data=np.array(
                                    [f[1] for f in failed],
                                    dtype=h5py.string_dtype(encoding="utf-8"),
                                ))

    print(f"\n[batch_calc] {n_ok}/{n} success, {len(failed)} failed, "
          f"{dt:.1f} s", flush=True)
    print(f"[batch_calc] wrote {out_path}", flush=True)

    if failed:
        with open(workdir / "failed.txt", "w") as f:
            for idx, msg in failed:
                f.write(f"mol_xyz_idx={idx}\t{msg}\n")
        print(f"[batch_calc] failure details in {workdir / 'failed.txt'}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="aimnet2-batch-calc",
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

    run_batch(cfg)


if __name__ == "__main__":
    main()
