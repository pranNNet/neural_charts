# Training preprocessing

This folder turns the **raw simulation file** into the **training file**. It runs no simulations. It only reads the raw file, keeps part of it, and adds two things derived from it: jump flags and the exact solution.

```
euler1d-data/data/step1_sod_pr_raw.h5  ──►  python -m preprocessing.build_dataset  ──►  preprocessing/data/step1_sod_pr.h5
        (input: raw)                               (this folder)                         (output: training file)
```

---

## 1. Running it

From `/mnt/project-files`:

```
python -m preprocessing.build_dataset                                   # paths and settings from preprocessing_config.yaml
python -m preprocessing.build_dataset --raw <raw.h5> --out <train.h5>   # other input and output files
python -m preprocessing.build_dataset --config <other.yaml>             # other settings
```

The settings live in `preprocessing_config.yaml`: the input and output paths (relative to the YAML file), the number of worker processes (`workers`, null = all cores), `T_train`, and the thresholds for the jump flags. A copy of the YAML is saved inside the output file. The full set of 104 cases should take under a minute on 4 cores (each case takes about 1 s).

| File | What it does |
|---|---|
| `preprocessing_config.yaml` | All the settings. |
| `build_dataset.py` | Reads each case from the raw file, runs the steps in §2, writes the training file and prints a check for each case. |
| `jump_flags.py` | Finds the cell faces where the data jumps (§2, step 3). |
| `exact.py` | Computes the exact solution as cell averages (§2, step 4). |
| `__init__.py` | Lets the package use two things from `euler1d-data/`: the conservative-to-primitive conversion and the exact Riemann solver. This is the only link between the two folders. |

---

## 2. What it does to the raw data

The raw file stores, for each of the 104 cases, the solution at **every solver step**: `U_fv_steps` [S+1, 512, 3], conservative variables (ρ, ρv, E), with S between about 2,500 and 5,300 depending on the case. For each case, preprocessing does four things:

1. **Keeps 121 snapshots.** It keeps only the steps at t = 0, 0.002, …, 0.24, using the raw file's `store_idx`. These are actual solver steps, not interpolated. [S+1, 512, 3] becomes [121, 512, 3].
2. **Converts to primitives.** (ρ, ρv, E) becomes (ρ, v, p), with v = ρv/ρ and p = (γ − 1)(E − ½ρv²). The result is `W_fv` [121, 512, 3]. The values are **not** rescaled, normalized or smoothed: the data is already dimensionless, so the fields are used exactly as the solver produced them. The only change is that they are stored as float32.
3. **Computes jump flags.** On each snapshot it looks at each of the 511 faces between neighbouring cells and measures the jump across it:
   - g_p = |Δp| / min(p) and g_ρ = |Δρ| / min(ρ) on the two sides, and g_v = Δv / (mean sound speed), which is negative when the flow is compressed.
   - A face is a **shock face** if g_p > 0.02 and g_v < −0.005 (a pressure jump with compression).
   - A face is a **contact face** if g_ρ > 0.02 and g_p < 0.005 (a density jump at nearly constant pressure).
   - The solver smears each jump over a few cells, so a jump shows up as a run of adjacent faces. Only **the face with the largest jump in each run** is flagged, so one jump gives exactly one flag.
   - The rarefaction fan is smooth, so it is never flagged.
   - Snapshots with **t < 0.02** are set to −1 ("ignore"), because the waves are still too close together there to separate. This covers t = 0 to 0.018, the first 10 snapshots.

   The result is `jump_flags` [121, 511]. In every snapshot that is used you expect exactly **2 flags**, one at the contact and one at the shock.
4. **Computes the exact solution.** From the case's left and right states it solves the Riemann problem exactly and averages the exact solution over each cell. The result is `U_exact` [121, 512, 3], in conservative variables, for **evaluation only**. It is never used for training.

The grid, the splits and every attribute are copied from the raw file unchanged.

**What is dropped:** all the in-between solver steps, `t_steps`, `store_idx` and the conservative FV fields. If you need any of them, they are still in the raw file.

**The check it prints.** For each case it prints the fewest and most flags on any used snapshot (expect 2 and 2), and the largest distance, in cells, between a flagged face and the exact contact or shock. On r = 2, 10, 13 and 16 every snapshot has exactly 2 flags, all within 0.72 cells of the exact front.

---

## 3. The output file

One HDF5 file, `preprocessing/data/step1_sod_pr.h5`, about 0.16 MB per case.

### Shared by all cases

| Path | Shape | Type | What it is |
|---|---|---|---|
| `/grid/x_edges` | [513] | float64 | cell edges, 0 to 1 |
| `/grid/x_centers` | [512] | float64 | cell centres; Δx = 1/512 |
| `/grid/t` | [121] | float64 | snapshot times 0, 0.002, …, 0.24 |
| `/splits/train` | [64] | int | case ids in the training set |
| `/splits/val` | [8] | int | validation |
| `/splits/test_id` | [16] | int | test, r inside the training range [2, 10] |
| `/splits/test_ood` | [16] | int | test, r outside it, (10, 16] |

File attributes: everything from the raw file (γ, x0, grid sizes, scheme, the generation YAML as `config_yaml`, …), plus `raw_file` (the input path), `T_train` = 0.16 and `preprocessing_config` (the preprocessing YAML used, as text).

### Per case: `/cases/000` … `/cases/103`

| Path | Shape | Type | What it is | Use |
|---|---|---|---|---|
| `W_fv` | [121, 512, 3] | float32 | FV solution, (ρ, v, p) cell averages | field target |
| `jump_flags` | [121, 511] | int8 | per face: 1 = jump, 0 = no jump, −1 = ignore | geometry target |
| `U_exact` | [121, 512, 3] | float32 | exact solution, (ρ, ρv, E) cell averages | evaluation only |

Attributes: `r` (pressure ratio, the encoder's input), `mu` = log r, `left` and `right` (the initial (ρ, v, p) states), `split`, `dt_solver` and `cfl_actual`.

### How to index

- **Array order is [time, cell, variable]**: `W_fv[j, i, k]` is variable k (0 = ρ, 1 = v, 2 = p) in cell i at time `t[j]`.
- **Faces:** `jump_flags[j, i]` is the face between cells i and i + 1, at x = `x_edges[i + 1]`.
- **Case ids** are the strings `"000"` to `"103"`. The split lists hold the same ids as integers, so use `f"{i:03d}"`.
- **Values are cell averages** over `[x_edges[i], x_edges[i+1]]`, not point values.
- **Time bands:** train and select models on t ≤ `T_train` = 0.16 (the first 81 snapshots). 0.16 < t ≤ 0.24 is held out for the later extrapolation-in-time test.
- **Normalized inputs** x̃ = 2x − 1 and t̃ = t / T_train are computed in your loader, not stored.

---

## 4. Reading it

One case:

```python
import h5py, numpy as np

f = h5py.File('preprocessing/data/step1_sod_pr.h5', 'r')
x  = f['grid/x_centers'][()]                 # [512]
xf = f['grid/x_edges'][1:-1]                 # [511]  face positions
t  = f['grid/t'][()]                         # [121]

case  = f['cases/000']
r     = case.attrs['r']                      # float
W     = case['W_fv'][()]                     # [121, 512, 3]  rho, v, p
flags = case['jump_flags'][()]               # [121, 511]     1 / 0 / -1
rho, v, p = W[..., 0], W[..., 1], W[..., 2]  # each [121, 512]

# shock and contact positions at t = 0.1
j = np.argmin(abs(t - 0.1))
print(xf[flags[j] == 1])                     # two positions
```

A whole split, as stacked arrays for training:

```python
T_train = f.attrs['T_train']
use = t <= T_train + 1e-12                                   # [121] bool, first 81 snapshots

ids   = [f'{i:03d}' for i in f['splits/train'][()]]
r     = np.array([f['cases'][c].attrs['r'] for c in ids])    # [64]
W     = np.stack([f['cases'][c]['W_fv'][use] for c in ids])  # [64, 81, 512, 3]
flags = np.stack([f['cases'][c]['jump_flags'][use] for c in ids])   # [64, 81, 511]

x_tilde = 2 * x - 1                                          # [512]
t_tilde = t[use] / T_train                                   # [81]
face_used = flags >= 0                                       # mask for the geometry loss
```

As PyTorch tensors: `torch.from_numpy(W)` gives a float32 tensor of the same shape, and `torch.from_numpy(flags).long()` gives the flags. The whole training split is about 35 MB in memory, so it fits easily.
