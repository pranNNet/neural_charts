import torch
import torch.nn as nn

from model.geometry import LevelSetsNet, front_frame, front_distances, advect, advect_scores   # AUDIT 3, 1 fix
from model.region import FieldNet, positive_clamp
from model.shock_interface import interface_speed, constrain_shocks

# ======================================================================================
# Helpers
# ======================================================================================

def sample_grid(W_grid, rows, u):
    """Input field at x~ positions u [P, 1] of batch rows [P], linear between cell centres and clamped
    to the end cells. W_grid [B, C, N_grid] -> [P, C]."""
    N = W_grid.shape[-1]
    f = ((u[:, 0] + 1) * N / 2 - 0.5).clamp(0, N - 1)         # continuous cell index
    i0 = f.floor().long().clamp(max = N - 2)
    w = (f - i0)[:, None]
    return (1 - w) * W_grid[rows, :, i0] + w * W_grid[rows, :, i0 + 1]

def cell_offset(u, N):
    """delta = (u - nearest cell centre) / dx in [-0.5, 0.5], u [P, 1] in x~ -> [P, 1]. LOCAL FIELDS."""
    f = (u + 1) * N / 2 - 0.5                                  # continuous cell index
    return f - f.round()

def cut_blend(phi, W_regions, n, B):
    """CUT BLEND (replaces the hard argmax blend). For queries at the n cell centres of each item, in order
    (points() in train/helper.py, and the stepper): a cell the front passes through takes the two zones' fields
    by volume fraction, so a sub-cell front survives into W instead of being rounded to a cell face (which made
    the next step's geometry restart from the face, and the contact run ahead ~0.77 cells/step).

    Per face between centres i, i+1 with labels a != b: t = where psi = phi_a - phi_b crosses 0, linear between
    the centres (zero at x_i + t dx). t >= 0.5: cell i+1 is (t - 0.5) zone a, the rest zone b; t < 0.5: cell i is
    (0.5 - t) zone b, the rest zone a. t is computed from detached scores: the field loss cannot move the
    geometry through the blend. Every other cell, and any query set that is not B rows x n centres, keeps the hard
    blend. phi [P, K], W_regions [P, K, F] -> (W [P, F], labels [P], weights [P, K])."""
    K = phi.shape[-1]
    labels = phi.argmax(dim=-1)
    weights = nn.functional.one_hot(labels, K).to(W_regions)                  # [P, K]
    if len(phi) == B * n:
        p, w, lab = phi.detach().view(B, n, K), weights.view(B, n, K).clone(), labels.view(B, n)
        a, b = lab[:, :-1], lab[:, 1:]                                         # zones either side of each face
        g = lambda P, z: P.gather(-1, z[..., None])[..., 0]
        si = g(p[:, :-1], a) - g(p[:, :-1], b)                                 # psi at centre i   (> 0 at a front)
        sj = g(p[:, 1:], a) - g(p[:, 1:], b)                                   # psi at centre i+1 (< 0 at a front)
        den = si - sj
        ok = den.abs() > 1e-12
        t = torch.where(ok, si / torch.where(ok, den, torch.ones_like(den)), torch.full_like(den, 0.5)).clamp(0, 1)
        front = a != b
        oa = nn.functional.one_hot(a, K).to(w)
        ob = nn.functional.one_hot(b, K).to(w)
        f1, k1 = (t - 0.5).clamp(min=0)[..., None], (front & (t >= 0.5))[..., None]
        w[:, 1:] = torch.where(k1, f1 * oa + (1 - f1) * ob, w[:, 1:])          # zero right of the face: cell i+1
        f0, k0 = (0.5 - t).clamp(min=0)[..., None], (front & (t < 0.5))[..., None]
        w[:, :-1] = torch.where(k0, f0 * ob + (1 - f0) * oa, w[:, :-1])        # zero left of the face: cell i
        weights = w.view(B * n, K)
    return (weights.unsqueeze(-1) * W_regions).sum(dim=1), labels, weights

# ======================================================================================
# NeuralChart step map: W(t_n) -> W(t_n+1).  GEOMETRY -> FIELD -> JUMP (speed, advect, constrain) -> BLEND
# ======================================================================================

class NeuralChart1DST(nn.Module):

    def __init__(self, config: dict, 
                 N_input = None):
        super().__init__()

        N_input = N_input or config['architecture']['coordinate_input']
        assert N_input == 1, "local model: sample_grid and the Conv1d features are 1D"

        #parameters for ablations
        self.geo_mode = config['architecture']['ablation']['geometry'].lower()
        self.jump_mode = config['architecture']['ablation']['jump_mode'].lower()
        self.head = config['architecture']['ablation']['head']

        assert self.geo_mode in ("levelset", "none")
        assert self.jump_mode in ("rh", "free") and self.head in ("softplus", "linear")

        geo_cfg, field_cfg = config['geometry'], config['field']
        self.N_grid = config['architecture']['N_grid']
        C = field_cfg['fieldnet']['field_output']

        # LOCAL FIELDS: one normalization for both CNNs, set once from the training data; saved with the checkpoint
        self.register_buffer("w_mean", torch.zeros(C, 1))
        self.register_buffer("w_std", torch.ones(C, 1))
        self.feat_grid = None            # geometry features [B, N_feat, N_grid], set once per step
        self.g_grid = None               # field features [B, C_f, N_grid], set once per step

        # FIELD module: the same local FieldNet in every arm; K = 1 for one-region (local CNN + INR)
        fcnn = field_cfg['feature_cnn']
        self.FieldNet = FieldNet(N_channels = C,
                                 N_regions = field_cfg['fieldnet']['regions'] if self.geo_mode == 'levelset' else 1,
                                 N_e_field = field_cfg['fieldnet']['N_e'] if self.geo_mode == 'levelset' else 1,
                                 hidden_layers = field_cfg['fieldnet']['hidden_layers'],
                                 N_out = C,
                                 N_bases = field_cfg['fieldnet']['n_bases'],
                                 N_in = N_input,
                                 conv_channels = fcnn['conv_channels'],
                                 N_feat = fcnn['N_feat'],
                                 kernel = fcnn['kernel'])

        if self.geo_mode == "levelset":
            gcnn = geo_cfg['feature_cnn']
            self.LevelSetsNet = LevelSetsNet(N_channels = C,
                                            N_out = geo_cfg['geometrynet']['phi_output'],
                                            conv_channels = gcnn['conv_channels'],
                                            N_feat = gcnn['N_feat'],
                                            kernel = gcnn['kernel'],
                                            hidden_layers = geo_cfg['geometrynet']['hidden_layers']
                                            )

            # shock constraint (JUMP module)
            jump_cfg, front_cfg = config['physics']['jump'], config['physics']['fronts']
            dx = 2 / self.N_grid                                     # cell size in x~: config lengths are in cells
            self.shock_cfg = {
                'gamma': config['physics']['constants']['gamma'],
                'tau_p': front_cfg['tau_p'],                         # pressure-rise threshold
                'Mn_gate': jump_cfg.get('Mn_gate', 1.05),            # Lax gate
                'side_offset': front_cfg['side_offset'] * dx,        # x~, plateau read distance
                'width': jump_cfg['width'] * dx,                     # x~, full-strength constraint
                'taper': jump_cfg['taper'] * dx,                     # x~, cosine taper after width
            }
            self.speed_rel_tol = jump_cfg['speed_rel_tol']           # interface_speed fallback
            self.dt_x = 2 * config['physics']['time']['dt'] / config['physics']['scaling']['Lx']   # x~ per unit speed per step
            assert self.shock_cfg['width'] >= self.shock_cfg['side_offset'], "pin point must lie inside width"
            assert self.shock_cfg['taper'] > 0
        else:
            self.LevelSetsNet = None

    def standardize(self, W_grid):
        return (W_grid.to(torch.float32) - self.w_mean) / self.w_std

    def set_features(self, W_grid):
        """The conv features of the input field W_grid [B, C, N_grid], once per step: g_grid for the fields and,
        in the level-set arms, feat_grid for the geometry. Every geo_phi / field_at call reads them."""
        W_std = self.standardize(W_grid)
        self.g_grid = self.FieldNet.feat_grid(W_std)
        self.feat_grid = self.LevelSetsNet.feat_grid(W_std) if self.LevelSetsNet is not None else None

    def geo_phi(self, u, rows):
        """u [P, D] normalized space, rows [P] or [P, 1] each point's row in W_grid -> phi [P, K].
        phi = LevelSetsNet(f(u)), f the features at u, linear between cell centres: the gradient in u
        (Newton normals) flows through the interpolation. Needs set_features first. Same call as in forward;
        front_distances passes rows as [P, 1]."""
        rows = rows.reshape(-1).long()
        return self.LevelSetsNet(sample_grid(self.feat_grid, rows, u))

    def field_at(self, u, rows):
        """Zone fields W_regions [P, K, C] at u [P, 1] of batch rows [P], heads applied. Needs set_features.
        The same local read for the cell centres and the jump layer's pins."""
        g = sample_grid(self.g_grid, rows.reshape(-1).long(), u)
        return self.heads_fn(self.FieldNet(cell_offset(u, self.N_grid), g))

    def heads_fn(self, h):
        """S.6, switched by `heads`. linear: rho = h1, p = h3 (positivity ablation, §6)."""
        return positive_clamp(h) if self.head == "softplus" else h

    def geometry(self, x, W_grid, group):
        """GEOMETRY at t_n: raw scores phi [N, K] from the input field W_grid [B, C, N_grid] (group [N] each
        point's row), detached unless the geometry trains. Used alone in stage 1 (sets the features).
        The owner of a point is phi.argmax(-1); phi is not a distance (front_frame gives those)."""
        train_geo = torch.is_grad_enabled() and any(p.requires_grad for p in self.LevelSetsNet.parameters())
        self.feat_grid = self.LevelSetsNet.feat_grid(self.standardize(W_grid))
        phi = self.geo_phi(x, group)                                       # AUDIT 3: plain forward
        return phi if train_geo else phi.detach()

    def jump_step(self, x, rows, phi, W_regions, W_grid, train_geo):
        """JUMP module. Band points get their front frame at t_n, the front speed from the input field,
        then advection to t_n+1 and the shock constraint there. 'free' runs it for the hooks but
        changes nothing. Returns (W_regions, jump, phi_next)."""
        cfg = self.shock_cfg
        reach = cfg['width'] + cfg['taper']
        d = phi.detach().argmax(dim=-1)                         # own zone

        # AUDIT 1 fix: j is the zone across the nearest front that is there (label check built in), chosen
        # by distance. The score rank says nothing about which zone is adjacent. Band points at t_n: a front
        # moves at most reach per step (advect), so dist <= 2 * reach can end within reach at t_n+1
        near, zones = front_distances(self.geo_phi, x, rows[:, None], d, cfg['side_offset']).topk(2, dim=-1, largest=False)
        j = zones[:, 0]
        idx = torch.nonzero(near[:, 0] <= 2 * reach).squeeze(1)
        rows_band = rows[idx]                                  # LOCAL GEOMETRY: rows in place of z_geo
        frame = front_frame(lambda u: self.geo_phi(u, rows_band), x[idx], d[idx], j[idx], train_geo)

        # time-n reads at the pin (own side) and the other side
        P = len(idx)
        step = cfg['side_offset'] * frame['n']
        u = torch.cat([frame['x_foot'] - step, frame['x_foot'] + step]).clamp(-1, 1)    # [2P, D]
        u = u.detach()                                     # reads only: no gradient through where they sit
        W_in = sample_grid(W_grid, rows[idx].repeat(2), u)
        W_in_d, W_in_j = W_in[:P], W_in[P:]

        # speed from the input field, then advect the fronts and the scores to t_n+1
        s = interface_speed(W_in_d, W_in_j, frame['n'], self.speed_rel_tol)
        frame_next, d_next, j_next, s_next, flipped = advect(frame, d[idx], j[idx], s, self.dt_x, reach)
        phi_next = advect_scores(phi, idx, d[idx], j[idx], frame_next['shift'] * frame['norm'])
        W_in_d, W_in_j = (torch.where(flipped[:, None], W_in_j, W_in_d),     # seen from the new owner
                          torch.where(flipped[:, None], W_in_d, W_in_j))

        # constrain only band points within reach at t_n+1 and not at a junction (label check is in front_distances)
        junction = near[idx, 1] <= reach                       # AUDIT 2: 2nd-nearest front within reach
        keep = (frame_next['dist'].detach() <= reach) & ~junction

        def fields_at(v, r):                                 # LOCAL FIELDS: the same local read as the cells
            return self.field_at(v, rows[r])

        W_regions, jump = constrain_shocks(W_regions, idx[keep], d_next[keep], j_next[keep],
                                           {key: val[keep] for key, val in frame_next.items()}, s_next[keep],
                                           (W_in_d[keep], W_in_j[keep]), fields_at, cfg,
                                           apply=self.jump_mode == 'rh')
        jump.update(flipped=flipped[keep], junction_frac=junction.float().mean())
        return W_regions, jump, phi_next

    def forward(self, x, W_grid, group) -> dict:     # LOCAL FIELDS: no latents

        '''
        INPUT REQUIREMENTS
        - x [N, D] normalized cell centres (space only), where W at t_n+1 is predicted. The cut blend needs all
          N_grid centres of each row, in order, rows in order (points()); any other query set gets the hard blend
        - W_grid [B, C, N_grid] the input field at t_n (primitives); group [N] each point's row in W_grid
          Both CNNs run once on W_grid; everything else reads their features locally.

        OUTPUT DESCRIPTION (all at t_n+1 unless marked):
        - W : Field Variables at coordinate
        - W_regions : What each regions predicts for the Field Variables
        - labels : the region the point is assigned too
        - phi_hat : geometry's raw score phi for each region (advected);  phi_hat_now : the same at t_n
        - weights : blend weights: one-hot from the labels, volume fractions in cut cells (cut_blend)
        - jump : per band point hooks from the shock constraint (see constrain_shocks), None for 'none'
        '''

        N = x.shape[0]
        self.set_features(W_grid)

        # for the BASELINE MODEL: the same local FieldNet, K = 1
        if self.geo_mode == 'none':
            W_regions = self.field_at(x, group)
            labels = x.new_zeros(N, dtype=torch.long)
            return {'W': W_regions[:, 0],
                    'W_regions': W_regions,
                    'labels': labels,
                    'phi_hat': None,
                    'phi_hat_now': None,
                    'weights': x.new_ones(N, 1),
                    'jump': None}

        # for the LEVELSET MODEL

        #forward GEOMETRY module, at t_n
        train_geo = torch.is_grad_enabled() and any(p.requires_grad for p in self.LevelSetsNet.parameters())
        phi_now = self.geo_phi(x, group)
        phi_now = phi_now if train_geo else phi_now.detach()

        #forward FIELD module, at t_n+1
        W_regions = self.field_at(x, group)

        #forward JUMP module: front speed, advection to t_n+1, pinned RH constraint behind shocks
        W_regions, jump, phi = self.jump_step(x, group, phi_now, W_regions, W_grid, train_geo)

        #BLEND (assembly), cut, with the advected scores: volume fractions in the cells the fronts pass through
        W, labels, weights = cut_blend(phi, W_regions, W_grid.shape[-1], W_grid.shape[0])

        return {'W': W,
                'W_regions': W_regions,
                'labels': labels,
                'phi_hat': phi,
                'phi_hat_now': phi_now,
                'weights': weights,
                'jump': jump}