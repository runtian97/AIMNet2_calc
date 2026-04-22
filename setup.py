from setuptools import setup, find_packages

setup(
    name="aimnet2calc",
    version="0.1.0",
    author="Roman Zubatyuk, yuyangwu (batch additions)",
    description="AIMNet2 calculators + pysisyphus batch runner with GPU batching",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "aimnet2calc": ["templates/*.yml", "templates/README.md"],
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
            # original single-reaction entry point
            "aimnet2pysis=aimnet2calc.aimnet2pysis:run_pysis",
            # batch driver: one YAML → parallel workers + BatchGPUServer
            "aimnet2-batch=aimnet2calc.batch_run:main",
        ],
    },
)
