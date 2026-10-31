import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


# ======================================================================================
# SnapshotDataset
# ======================================================================================

class SnapshotDataset(Dataset):
    '''
    Every snapshot with 0 < t <= t_max (default T_train) of every case in one split.
    Fields are as stored. Only x and t are mapped: x~ = 2(x - x_min)/L - 1, t~ = t/T_train.

    Shared:    x_edges [513], x_centers [512], x_faces [511] (all x~), t [S] (t~), case_ids
    Per item:  W [512, 3], flags [511] (1 / 0 / -1), t (t~), r [1], case, snap
               (+ U_exact [512, 3], conservative, if exact=True)
    '''

    def __init__(self, path, split, t_max=None, exact=False):
        with h5py.File(path, 'r') as f:
            T_train = float(f.attrs['T_train'])
            edges, t = f['grid/x_edges'][()], f['grid/t'][()]
            use = (t > 0) & (t <= (t_max or T_train) + 1e-12)       # t = 0 is the initial state, not used

            self.case_ids = [f'{i:03d}' for i in f[f'splits/{split}'][()]]
            cases = [f['cases'][c] for c in self.case_ids]
            stack = lambda key: torch.from_numpy(np.stack([c[key][()][use] for c in cases]))

            self.r = torch.tensor([c.attrs['r'] for c in cases], dtype=torch.float32)   # [C]
            self.W = stack('W_fv')                                                     # [C, S, 512, 3]
            self.flags = stack('jump_flags').long()                                    # [C, S, 511]
            self.U_exact = stack('U_exact') if exact else None                         # [C, S, 512, 3]

        self.x_edges = torch.from_numpy(2 * (edges - edges[0]) / (edges[-1] - edges[0]) - 1).float()
        self.x_centers = 0.5 * (self.x_edges[1:] + self.x_edges[:-1])
        self.x_faces = self.x_edges[1:-1]
        self.t = torch.from_numpy(t[use] / T_train).float()
        self.C, self.S = self.W.shape[:2]
        self.use, self.T_train = use, T_train                      # RolloutDataset reuses these

    def __len__(self):
        return self.C * self.S

    def __getitem__(self, i):
        c, j = divmod(i, self.S)
        item = {'W': self.W[c, j], 'flags': self.flags[c, j], 't': self.t[j],
                'r': self.r[c:c + 1], 'case': c, 'snap': j}
        if self.U_exact is not None:
            item['U_exact'] = self.U_exact[c, j]
        return item


# ======================================================================================
# RolloutDataset
# ======================================================================================

class RolloutDataset(SnapshotDataset):
    '''
    Items (case, n) with t_n >= t0 and t_{n+R} <= t_max (default T_train). R = None: one item per
    case, from t0 to t_max (the full rollout, for validation and test). R = 1: one-step pairs.

    Per item:  W_in [512, 3] at n; W_out [R, 512, 3] at n+1 .. n+R;
               classes [R+1, 512] (-1 = ignored) and flags [R+1, 511] at n .. n+R; r [1], case, n
    '''

    def __init__(self, path, split, R=1, t0=0.02, t_max=None):
        super().__init__(path, split, t_max)
        with h5py.File(path, 'r') as f:
            self.classes = torch.from_numpy(np.stack([f['cases'][c]['classes'][()][self.use]
                                                      for c in self.case_ids])).long()  # [C, S, 512]

        self.n0 = int((self.t < t0 / self.T_train - 1e-9).sum())     # first snapshot with t >= t0
        self.R = R or self.S - self.n0 - 1
        self.N = self.S - self.n0 - self.R                           # start times per case

    def __len__(self):
        return self.C * self.N

    def __getitem__(self, i):
        c, k = divmod(i, self.N)
        n = self.n0 + k
        return {'W_in': self.W[c, n], 'W_out': self.W[c, n + 1:n + self.R + 1],
                'classes': self.classes[c, n:n + self.R + 1], 'flags': self.flags[c, n:n + self.R + 1],
                'r': self.r[c:c + 1], 'case': c, 'n': n}


def load_splits(path, cls=SnapshotDataset, **kw):
    '''(train, val, test_id, test_ood). kw goes to every split, e.g. exact=True or R=4.'''
    return tuple(cls(path, s, **kw) for s in ('train', 'val', 'test_id', 'test_ood'))
