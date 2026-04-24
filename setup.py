from setuptools import setup, find_packages

setup(
    name="aimnet2calc",
    version="0.1.0",
    author="Roman Zubatyuk, yuyangwu (batch additions)",
    description="AIMNet2 calculators + pysisyphus batch runner with GPU batching",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "aimnet2calc": [
            "templates/README.md",
            "templates/*/*.yml",
            "templates/*/.gitkeep",
        ],
    },
    install_requires=[
        # Core PyTorch stack — pin range we've tested on.
        "torch>=2.4,<2.6",
        "torch_cluster>=1.6,<1.7",
        # Numeric stack
        "numpy>=1.26,<2",
        "numba>=0.60",
        "scipy>=1.14,<2",
        # I/O + chemistry
        "ase>=3.23",
        "h5py>=3.11",
        "pyyaml>=6",
        # batch_run / finalize use these
        "requests>=2.31",
        # pysisyphus must be installed separately (from git); see README
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    entry_points={
        "console_scripts": [
            # one YAML → many reactions through pysisyphus pipeline
            # (preopt → NEB → TSopt → IRC → endopt), sharing one GPU via
            # BatchGPUServer that coalesces force / hessian requests
            "aimnet2-batch-pysis=aimnet2calc.batch_run:main",
            # one YAML → many molecules through ASE LBFGS geometry opt,
            # cross-molecule batching via the same BatchGPUServer
            "aimnet2-batch-geom=aimnet2calc.batch_geom_opt:main",
            # one YAML → single-point energy / forces / hessian (+ optional
            # thermal H/S/G) on many molecules, one batched forward per
            # atom-count group
            "aimnet2-batch-calc=aimnet2calc.batch_calc:main",
        ],
    },
)
