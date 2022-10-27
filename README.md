# Automatic Differentiation for the Direct Minimization Approach to the Hartree-Fock Method
Source code for [Automatic Differentiation for the Direct Minimization Approach to the Hartree-Fock Method](https://doi.org/10.1021/acs.jpca.2c05922.) ([arXiv](https://arxiv.org/abs/2203.04441)).

## Dependencies
- JAX
- PySCF
- SciPy
- NumPy
- matplotlib

You can install dependencies with `requirements.txt`.

```
pip install -r requirements.txt
```

## How to reproduce Figures/Tables
We tested our program with Python 3.8.10 on Ubuntu 20.04.5 LTS.

### Figure 2(a)
```
python diatomic.py H2 STO-3G
```

### Figure 2(b)
```
python diatomic.py H2 3-21G
```

### Figure 3(a)
```
python diatomic.py HF STO-3G
```

### Figure 3(b)
```
python diatomic.py HF STO-3G
```

### Figure 4(a)
```
python polyatomic_angle.py H2O
```

### Figure 4(b)
```
python polyatomic_angle.py NH3
```

### Figure 5
```
python ad_vs_fd.py NH3 cc-pVDZ --plot
```

### Figure 6
```
python plot_step_size.py
```

`data/step_size_h2o.csv` was created by running `ad_vs_fd.py` in different step sizes (change line 173).

### Table 3
You can calculate each row using `polyatomic.py`. For example, 
```
python polyatomic.py H2O STO-3G
```

### Table 4
You can calculate each row using `ad_vs_fd.py`. For example, 
```
python ad_vs_fd.py H2O STO-3G
```
