"""MedMNIST adapter for the frozen plaintext conversion harness.

harness/pvit_lab/ is a frozen 2026-09-17 reference (see
paper/analysis_20260920/PROVENANCE.md) and is NOT modified by this package.
This package supplies a drop-in replacement for pvit_lab.data.datasets and
installs it by assignment at call time.
"""
from .data import datasets, SPECS  # noqa: F401
