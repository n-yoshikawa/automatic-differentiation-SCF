# Automatic differentiation for the direct SCF approach to the Hartree-Fock method
Source code for [Automatic differentiation for the direct SCF approach to the Hartree-Fock method](https://arxiv.org/abs/2203.04441).

## Requirements
- JAX
- PySCF
- SciPy
- NumPy
- matplotlib

## Files
- `adscf/` - Library for automatic differentiation of HF energy
- `diatomic.py` - Draw energy curve for diatomic molecules
- `polyatomic.py` - Calculate energy for polyatomic molecules
- `ad_vs_fd.py` - Compare automatic differentiation and numerical differentiation
- `diatomic_julia.jl` - Implementation in Julia

After installing the dependent libraries, just run
```
python diatomic.py
```
