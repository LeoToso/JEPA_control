"""
Setup script for the JEPA Control research codebase.
"""

from setuptools import setup, find_packages

setup(
    name="jepa_control",
    version="0.1.0",
    description=(
        "Control-theoretic probing of JEPA encoders on the cartpole system"
    ),
    author="Research",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "gymnasium>=0.29.0",
        "pygame>=2.1.0",
        "h5py>=3.8.0",
        "PyYAML>=6.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": ["pytest>=7.3.0"],
        "cv": ["opencv-python>=4.7.0"],
        "wandb": ["wandb>=0.15.0"],
    },
    entry_points={
        "console_scripts": [
            "jepa-run=experiments.run_experiment:main",
            "jepa-grid=experiments.run_all:main",
        ],
    },
)
