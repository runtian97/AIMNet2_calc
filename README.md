# AIMNet2 Calculator: Fast, Accurate Molecular Simulations

> **Model compatibility:** this package expects a **JIT-compiled AIMNet2
> model** (a `.jpt` file loadable via `torch.jit.load`).  Pass the model
> path to `AIMNet2Calculator("/path/to/model.jpt")` or via the YAML
> `calc.model:` field.  Non-JIT (Python-only) AIMNet2 checkpoints are not
> supported.

This package integrates the powerful AIMNet2 neural network potential into your simulation workflows. AIMNet2 provides fast and reliable energy, force, and property calculations for molecules containing a diverse range of elements.

## Key Features:

- **Accurate and Versatile:** AIMNet2 excels at modeling neutral, charged, organic, and elemental-organic systems.
- **Flexible Interfaces:** Use AIMNet2 through convenient calculators for popular simulation packages like ASE and PySisyphus.
- **Flexible Long-Range Interactions:** Optionally employ the Dumped-Shifted Force (DSF) or Ewald summation Coulomb models for accurate calculations in large or periodic systems.


## Getting Started

### 1. Installation

Versions below are the ones we've verified on an NVIDIA A40/A100 node with
CUDA 12.4.  Copy-paste the whole block:

```bash
# 1. Fresh conda env with Python 3.11
conda create -y -n aimpysis python=3.11
conda activate aimpysis

# 2. PyTorch 2.5.1 + CUDA 12.4
conda install -y -c pytorch -c nvidia \
    pytorch=2.5.1 pytorch-cuda=12.4

# 3. pysisyphus runtime deps + chemistry / IO packages
conda install -y -c conda-forge \
    openbabel=3.1 ase=3.23 \
    numpy=1.26 scipy=1.14 scikit-learn=1.5 sympy=1.13 numba=0.60 \
    h5py=3.11 matplotlib=3.9 \
    pyyaml=6.0 rmsd=1.5 joblib=1.4 natsort=8.4 psutil=6.0 \
    jinja2=3.1 autograd=1.7 fabric=3.2 \
    dask=2024.9 distributed=2024.9 requests=2.32

# 4. torch_cluster wheel (conda build isn't published for torch 2.5.1)
pip install torch_cluster==1.6.3 \
    -f https://data.pyg.org/whl/torch-2.5.1+cu124.html

# 5. pysisyphus from git — pypi is stale, we need the 0.8.0b1 dev branch
pip install "git+https://github.com/eljost/pysisyphus.git@30880698"

# 6. This package in editable mode
pip install -e .
```

Pinned versions (what the commands above install):

| Package            | Version                        |
| ------------------ | ------------------------------ |
| python             | 3.11                           |
| pytorch            | 2.5.1 (cuda 12.4, cudnn 9.1)   |
| pytorch-cluster    | 1.6.3 (pip, from pyg wheel)    |
| openbabel          | 3.1                            |
| ase                | 3.23                           |
| numpy              | 1.26                           |
| scipy              | 1.14                           |
| scikit-learn       | 1.5                            |
| sympy              | 1.13                           |
| numba              | 0.60                           |
| h5py               | 3.11                           |
| matplotlib         | 3.9                            |
| pyyaml             | 6.0                            |
| rmsd               | 1.5                            |
| dask / distributed | 2024.9                         |
| pysisyphus         | 0.8.0b1.dev155+g30880698 (git) |

If your node has a different CUDA driver, change `pytorch-cuda=12.4` to match
(PyTorch 2.5.1 is available for CUDA 11.8, 12.1, and 12.4 on the pytorch
channel).  Also change the torch_cluster wheel URL accordingly
(e.g. `torch-2.5.1+cu121.html`).  All other pins stay the same.

Sanity check after install:

```bash
python -c "from aimnet2calc import AIMNet2Calculator; print('ok')"
python -c "from aimnet2calc.aimnet2pysis import AIMNet2Pysis, BatchGPUServer; print('ok')"
python -c "from aimnet2calc.batch_run import main; print('ok')"
```

### Entry points installed

| Command                 | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `aimnet2-batch-pysis`   | Batch pysisyphus pipeline: preopt → NEB → TSopt →    |
|                         | IRC → endopt for many reactions in a single YAML,    |
|                         | all sharing one GPU via BatchGPUServer.              |

`aimnet2-batch-pysis input.yml` is equivalent to
`python -m aimnet2calc.batch_run input.yml`.

| Command                 | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `aimnet2-batch-geom`    | Batch geometry optimisation (ASE LBFGS)              |
| `aimnet2-batch-calc`    | Batch single-point energy / forces / hessian         |
|                         | (+ optional thermal H / S / G via IdealGasThermo)    |

### 2. Available interfaces

#### ASE [[https://wiki.fysik.dtu.dk/ase]](https://wiki.fysik.dtu.dk/ase)

```
from aimnet2calc import AIMNet2ASE
calc = AIMNet2ASE('aimnet2')
```

To specify total molecular charge and spin multiplicity, use optional `charge` and `mult` keyword arguments, or  `set_charge` and `set_mult` methods:

```
calc = AIMNet2ASE('aimnet2', charge=1)
atoms1.calc = calc
# calculations on atoms1 will be done with charge 1
....
atoms2.calc = calc
calc.set_charge(-2)
# calculations on atoms1 will be done with charge -2
```

#### PySisyphus [[https://pysisyphus.readthedocs.io]](https://pysisyphus.readthedocs.io/)

```
from aimnet2calc import AIMNet2PySis
calc = AIMNet2PySis('aimnet2')
```

This produces standard PySisyphus calculator.

Instead of `Pysis` command line utility, use `aimnet2pysis`. This registeres AIMNet2 calculator with PySisyphus.
Example `calc` section for PySisyphus YAML files:

```
calc:
   type: aimnet              # use AIMNet2 calculator
   model: aimnet2_b973c      # use aimnet2_b973c_0.jpt model
```

### 3. Base calculator

```
from aimnet2calc import AIMNet2Calculator
```

#### Initialization

```
calc = AIMNet2Calculator('aimnet2')
```
will load default AIMNet2 model aimnet2_wb97m_0.jpt as defined at `aimnet2calc/models.py` . If file does not exist on the machine, it will be downloaded from [aimnet-model-zoo](http://github.com/zubatyuk/aimnet-model-zoo) repository.

```
calc = AIMNet2Calculator('/path/to_a/model.jpt')
```
will load model from the file. 

#### Input structure

The calculator accepts a dictionary containig lists, numpy arrays, torch tensors, or anything that could be accepted by `torch.as_tensor`. 

The input could be for a single molecule (dict keys and shapes):

```
coord: (B, N, 3)  # atomic coordinates in Angstrom
numbers (B, N)    # atomic numbers
charge (B,)       # molecular charge
mult (B,)         # spin multiplicity, optional
```

or for a concatenation of molecules:

```
coord: (N, 3)  # atomic coordinates in Angstrom
numbers (N,)    # atomic numbers
charge (B,)    # molecular charge
mult (B,)      # spin multiplicity, optional
mol_idx (N,)   # molecule index for each atom, should contain integers in increasing order, with (B-1) is the maximum number.
```

where `B` is the number of molecules, `N` is number of atoms. 


#### Calling calculator

```
results = calc(data, forces=False, stress=False, hessian=False)
```

`results` would be a dictionary of PyTorch tensors containing `energy`, `charges`, and possibly `forces`, `stress` and `hessian` if requested.

### 4. Long range Coulomb model

By default, Coulomb energy is calculated in O(N^2) manner, e.g. pair interaction between every pair of atoms in system. For very large or periodic systems, O(N) Dumped-Shifted Force Coulomb model could be employed [doi: 10.1063/1.2206581](https://doi.org/10.1063/1.2206581). With `AIMNet2Calculator` interface, switch between standard and DSF Coulomb implementations im AIMNet2 models:

```
# switch to O(N)
calc.set_lrcoulomb_method('dsf', cutoff=15.0, dsf_alpha=0.2)
# switch to O(N^2), not suitable for PBC
calc.set_lrcoulomb_method('simple')
```




