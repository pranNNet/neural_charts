
## 1. What the dataset is

The dataset is **104 shock-tube simulations**. They differ in **one number only**: the pressure ratio **r**.

Every case starts with a tube on x ∈ [0, 1] with a diaphragm at x = 0.4:

| | density ρ | velocity v | pressure p |
|---|---|---|---|
| left of 0.4 | 1 | 0 | 0.1 · r |
| right of 0.4 | 0.125 | 0 | 0.1 |

γ = 1.4. When the diaphragm bursts, every case develops the same pattern, from left to right:
1. a rarefaction fan moving left;
2. a contact (a density jump at equal pressure) moving right;
3. a shock moving right.

A larger r means a stronger, faster shock. r = 10 is the classic Sod problem.

Each simulation runs from t = 0 to t = 0.24. No wave reaches either end of the tube in that time.

**How the 104 values of r were chosen.** r is sampled evenly in log r, with no gaps, in four groups. The group names say what they are meant for later. The raw dataset only records which group each case belongs to.

| Group | Cases | r range |
|---|---|---|
| train | 64 | 2 to 10 |
| val | 8 | 2 to 10 |
| test_id | 16 | 2 to 10 |
| test_ood | 16 | 10 to 16 |

---

## 2. How it is made

Every setting lives in **`generation_config.yaml`**:
- the tube and the two states;
- the grid and times;
- the scheme;
- the values of r for each split;
- the output path.

Each setting is commented in the file. Change the YAML, not the code. Every script takes `--config` to use another file. The raw file stores a copy of the YAML it was made with, in the attribute `config_yaml`, so you can always see exactly how a file was produced.

Requires numpy, scipy, h5py, matplotlib and pyyaml.

```
python data_generation.py                                        # all 104 cases, about 25 min on 4 cores
python data_generation.py --limit 1 --out data/smoke_raw.h5      # one case per split, for a quick test
python data_generation.py --config other.yaml                    # a different setup
```

For each case:
1. Your solver (HLLC flux, WENO5Z reconstruction, RK4 time stepping) runs on **4096 cells**.
2. The time step dt is fixed for the case and chosen for CFL ≈ 0.5. It divides 0.002 exactly, so t = 0, 0.002, 0.004, … are actual solver steps.
3. **Every solver step is kept.** That is about 2,500 steps at r = 2 and about 5,300 at r = 16.
4. Before saving, each group of 8 fine cells is averaged into one, giving **512 cells**. This averaging is exact for cell averages, because mass, momentum and energy are conserved.

| Script | What it does |
|---|---|
| `generation_config.yaml` | All the settings. |
| `run.py` | Reads the config and runs **one** case for a given r (`python run.py --r 10` times one case). |
| `data_generation.py` | Picks the 104 values of r, runs every case, checks that mass and energy are conserved, and writes the raw file. |
| `Sod_Shock_Tube_Validation.py` | Checks the solver; it is not part of the dataset. It compares the exact Riemann solver with published values and your FV solver with the exact solution on Sod. It prints PASS or FAIL and writes `sod_validation.png`. |

---

## 3. What is saved

Two files:
- **`data/step1_sod_pr_raw.h5`**: the dataset.
- **`data/step1_sod_pr_raw_report.json`**: a run log. Per case it records dt, CFL, the number of steps, the run time and the mass and energy drift (expect about 1e-16).

### Layout of `data/step1_sod_pr_raw.h5`

```
/  (attributes)    config_yaml (the full YAML used), gamma = 1.4, x0 = 0.4, domain = [0, 1], Nx = 512, Nx_fine = 4096,
                   dt_store = 0.002, t_max = 0.24, solver, flux = "HLLC", reconstruction = "WENO5Z",
                   time_integrator, cfl = 0.5, coarsening, code_git_hash
/grid/x_edges      [513]       cell edges
/grid/x_centers    [512]       cell centres
/grid/t            [121]       0, 0.002, ..., 0.24
/splits/train      [64]        case ids (integers) in each group; also val [8], test_id [16], test_ood [16]
/cases/000 ... /cases/103      one group per case
    attributes:    r, mu = log r, left = (ρ, v, p), right = (ρ, v, p), split (group name),
                   dt_solver (this case's time step), cfl_actual
    U_fv_steps     [S+1, 512, 3]  float32   the solution at every solver step
    t_steps        [S+1]          float64   the time of each step
    store_idx      [121]          int       which steps are at t = 0, 0.002, ..., 0.24
```

S, the number of steps, differs from case to case, because dt does.

### What the numbers mean

- **`U_fv_steps[k, i, :]`** is the solution at time `t_steps[k]` in cell i.
- **The three variables are conservative**, in this order:
  - **ρ**: density;
  - **ρv**: momentum;
  - **E**: total energy, E = p/(γ−1) + ½ρv².
- **Each value is a cell average** over `[x_edges[i], x_edges[i+1]]`, not a point value. The cell centre is `x_centers[i]` and Δx = 1/512.
- **Array order is `[time, cell, variable]`.**
- **To get primitives (ρ, v, p):** ρ = U[..., 0], v = U[..., 1] / ρ, p = 0.4 · (U[..., 2] − ½ ρ v²).
- **`U_fv_steps[store_idx]`** gives the 121 states at t = 0, 0.002, …, 0.24, matching `/grid/t`.
- **Case ids** are zero-padded strings `"000"` to `"103"`. The `/splits` lists hold the same ids as integers (format them with `f"{i:03d}"`).

---

## 4. Working with it

```python
import h5py, numpy as np

f = h5py.File('data/step1_sod_pr_raw.h5', 'r')
x = f['grid/x_centers'][()]                       # [512]

ids = [f'{i:03d}' for i in f['splits/train'][()]]   # the 64 "train" cases
case = f['cases'][ids[0]]
r = case.attrs['r']

U = case['U_fv_steps'][()]                        # [S+1, 512, 3]  every step
t = case['t_steps'][()]                           # [S+1]

# primitives
rho = U[..., 0]
v   = U[..., 1] / rho
p   = 0.4 * (U[..., 2] - 0.5 * rho * v**2)

# only the 121 states on the 0.002 grid (reads just those steps from disk)
U121 = case['U_fv_steps'][case['store_idx'][()]]  # [121, 512, 3]

# all cases with their r
rs = {cid: f['cases'][cid].attrs['r'] for cid in f['cases']}
```
