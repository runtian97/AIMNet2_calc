# Ready-to-use batch_run templates

Five input-YAML templates for `python -m aimnet2calc.batch_run`, ranked by
measured PASS rate on a 30-reaction SN2 screen.  Each file is a complete,
self-contained input — copy it, edit the `r_xyz`/`p_xyz`/`model` paths and
charge/mult to match your dataset, then run:

    python -m aimnet2calc.batch_run <template>.yml

PASS criterion: IRC connectivity (endpoints map to R / P) AND exactly 1
imaginary frequency at the TS.

| File                    | coord  | cos   | tsopt  | irc     | PASS  | wall |
| ----------------------- | ------ | ----- | ------ | ------- | ----- | ---- |
| 01_recommended.yml      | cart   | szts  | rsirfo | eulerpc | 29/30 | 538s |
| 02_fast.yml             | cart   | neb   | rsirfo | eulerpc | 27/30 | 407s |
| 03_string_method.yml    | redund | szts  | rsirfo | eulerpc | 27/30 | 486s |
| 04_baseline.yml         | redund | neb   | rsirfo | eulerpc | 27/30 | 517s |
| 05_irc_rk4.yml          | redund | neb   | rsirfo | rk4     | 26/30 | 627s |

Defaults to start with:

  * unless you know better, use **01_recommended.yml**
  * use **02_fast.yml** when throughput matters more than the last 1-2 PASSes
  * use **03_string_method.yml** if NEB struggles on your barriers
  * use **04_baseline.yml** as a debugging reference (pysisyphus defaults)
  * use **05_irc_rk4.yml** only when eulerpc IRC gives unstable modes

Notes on the other options that were tested but didn't make the top 5:
  * `coord=dlc`               slightly lower PASS than cart/redund (25/30)
  * `cos=gs` (growing string) never converged on this SN2 set (0/30)
  * `tsopt=rsprfo`            slower, 1 PASS fewer than rsirfo (26/30)
  * `tsopt=trim`              many failures (11/30)
  * `irc=gs`                  many failures (17/30)

All templates use these reusable defaults, which you rarely need to change:

  * `opt.type: lbfgs`, `opt.align: True`, `rms_force: 0.01`, `max_step: 0.04`
  * `tsopt.do_hess: True`, `tsopt.hessian_recalc: 7`, `tsopt.thresh: gau_tight`
  * `interpol.type: redund`, `interpol.between: 10`
  * `endopt.fragments: True`
  * `batch.slim: true` (removes per-run intermediate files; keeps products up
    to the last successful stage)
