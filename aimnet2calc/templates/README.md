# Ready-to-use `aimnet2-batch-*` templates

Templates are grouped by the CLI / task that consumes them.  Copy the
template closest to your use case, edit the paths (and charge / mult) to
match your dataset, then run the matching `aimnet2-batch-*` command.

## Directory layout

```
templates/
├── batch_pysis/        ← pysisyphus reaction pipeline
│                         (preopt → NEB → TSopt → IRC → endopt)
│                         CLI:  aimnet2-batch-pysis
├── batch_geom_opt/     ← geometry optimisation (ASE LBFGS)
│                         CLI:  aimnet2-batch-geom
└── batch_calc/         ← single-point energy / forces / hessian, plus
                          optional thermal properties (H / S / G) via
                          IdealGasThermo in the same pass.
                          CLI:  aimnet2-batch-calc
```

## `batch_pysis/`

Five input-YAML templates for `aimnet2-batch-pysis`, ranked by measured PASS
rate on a 30-reaction SN2 screen (PASS = IRC connectivity OK AND TS has
exactly one imaginary frequency):

| File                    | coord  | cos   | tsopt  | irc     | PASS  | wall |
| ----------------------- | ------ | ----- | ------ | ------- | ----- | ---- |
| 01_recommended.yml      | cart   | szts  | rsirfo | eulerpc | 29/30 | 538s |
| 02_fast.yml             | cart   | neb   | rsirfo | eulerpc | 27/30 | 407s |
| 03_string_method.yml    | redund | szts  | rsirfo | eulerpc | 27/30 | 486s |
| 04_baseline.yml         | redund | neb   | rsirfo | eulerpc | 27/30 | 517s |
| 05_irc_rk4.yml          | redund | neb   | rsirfo | rk4     | 26/30 | 627s |

Start with `01_recommended.yml` unless you have a reason to pick another.

## `batch_geom_opt/`

Single template `geom_opt.yml` for `aimnet2-batch-geom`.  Takes a multi-frame
XYZ (one molecule per frame), runs ASE LBFGS on every molecule in parallel
sharing one GPU via `BatchGPUServer`, and sorts results into
`converged/` vs `not_converged/`.

## `batch_calc/`

Single template `calc.yml` for `aimnet2-batch-calc`.  Takes a multi-frame
XYZ, groups molecules by atom count, and does one batched AIMNet2 forward
per group.  Outputs a single `results.h5` keyed by atom count (`NNN/`
subgroups).

Properties toggled in the YAML:

| flag in `properties` | saved in each `NNN/` group   | cost     |
| -------------------- | ---------------------------- | -------- |
| `energy: true`       | `energy (B,) eV`             | baseline |
| `forces: true`       | `forces (B, N, 3) eV/Å`      | +10-20%  |
| `hessian: true`      | `hessian (B, 3N, 3N) eV/Å²`  | +5-10×   |

Optionally also compute H / S / G per molecule with ASE `IdealGasThermo`:

| `thermal.enabled`   | additional saved arrays            |
| ------------------- | ---------------------------------- |
| `true`              | `enthalpy (B,) eV`                 |
|                     | `entropy (B,) eV/K`                |
|                     | `gibbs_energy (B,) eV`             |
| `false` (or omitted)| none (skip thermal step)           |

The `thermal:` block also carries the parameters (temperature, pressure,
geometry, symmetry_number, ignore_imag_modes).  `thermal.enabled: true`
requires `properties.hessian: true`.

## Common options that apply to every CLI

Defaults to start with (rarely need to change):
  * `calc.type: aimnet` and `calc.model: …`
  * `batch.slim: true` — keeps only essential outputs per run, splits into
    success / fail subdirectories
  * `batch.max_workers: null` — defaults to running all items concurrently;
    set a finite cap (e.g. `50`) when you have hundreds of items to avoid
    CPU / RAM pressure
