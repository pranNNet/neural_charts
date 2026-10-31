import torch
import torch.nn as nn

from model.transforms import build_B, MSFourier_Features
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

# ======================================================================================
# NeuralChart step map: W(t_n) -> W(t_n+1).  GEOMETRY -> FIELD -> JUMP (speed, advect, constrain) -> BLEND
# ======================================================================================

class NeuralChart1DST(nn.Module):

    def __init__(self, config: dict, 
                 N_input = None):
        super().__init__()

        # number of coordinates, D+1 = (space..., t)
        N_input = N_input or config['architecture']['coordinate_input']

        #parameters for ablations
        self.geo_mode = config['architecture']['ablation']['geometry'].lower()
        self.jump_mode = config['architecture']['ablation']['jump_mode'].lower()

        self.fourier = config['architecture']['ablation']['fourier'] in (True, 'on')
        self.head = config['architecture']['ablation']['head']

        assert self.geo_mode.lower() in ("levelset", "none")
        assert self.jump_mode.lower() in ("rh", "free") and self.head in ("softplus", "linear")

        geo_cfg = config['geometry']
        field_cfg = config['field']

        # Key diff is Level-set or Not
        if self.geo_mode == "levelset":

            #bands for multi-scale fourier features
            self.B_geo = build_B(bands = geo_cfg['fourierfeatures']['bands'],
                                 seed = geo_cfg['fourierfeatures']['seed'],
                                 fourier=self.fourier,
                                 N_input = N_input
                                 )

            self.B_field = build_B(bands = field_cfg['fourierfeatures']['bands'],
                                   seed = field_cfg['fourierfeatures']['seed'],
                                   fourier=self.fourier,
                                   N_input = N_input
                                   )

            self.LevelSetsNet = LevelSetsNet(N_in = N_input + 2*self.B_geo.shape[1],
                                            z_geo = config['architecture']['latent']['z_geo'],
                                            N_out = geo_cfg['geometrynet']['phi_output'],
                                            hidden_layers = geo_cfg['geometrynet']['hidden_layers']
                                            )

            self.FieldNet = FieldNet(N_in = N_input + 2*self.B_field.shape[1],
                                     N_regions = field_cfg['fieldnet']['regions'],
                                     N_z_field = config['architecture']['latent']['z_field'],
                                     N_e_field = field_cfg['fieldnet']['N_e'],
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers'],
                                     N_out = field_cfg['fieldnet']['field_output'],
                                     N_bases = field_cfg['fieldnet']['n_bases']
                                     )
            # shock constraint (JUMP module)
            jump_cfg, front_cfg = config['physics']['jump'], config['physics']['fronts']
            dx = 2 / config['encoder']['N_grid']                    # cell size in x~: config lengths are in cells
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
            # Baseline comparison (no Level Set Functionality)

            #bands for multi-scale fourier features
            self.B_geo, self.LevelSetsNet = None, None

            self.B_field = build_B(bands = field_cfg['fourierfeatures']['bands'],
                                   seed = field_cfg['fourierfeatures']['seed'],
                                   fourier=self.fourier,
                                   N_input = N_input
                                   )

            self.FieldNet = FieldNet(N_in = N_input + 2*self.B_field.shape[1],
                                     N_regions = 1,
                                     N_z_field = config['architecture']['latent']['z_field'] + config['architecture']['latent']['z_geo'],
                                     N_e_field = 1,
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers'],
                                     N_out = field_cfg['fieldnet']['field_output'],
                                     N_bases = field_cfg['fieldnet']['n_bases']
                                     )

    def geo_phi(self, u, z):
        """u [P, D] normalized space, z [P, z_geo] -> phi [P, K]. Same call as in forward."""
        return self.LevelSetsNet(MSFourier_Features(u, self.B_geo), z)

    def heads_fn(self, h):
        """S.6, switched by `heads`. linear: rho = h1, p = h3 (positivity ablation, §6)."""
        return positive_clamp(h) if self.head == "softplus" else h

    def geometry(self, x, z_geo):
        """GEOMETRY at t_n: raw scores phi [N, K], detached unless the geometry trains. Used alone in stage 1.
        The owner of a point is phi.argmax(-1); phi is not a distance (front_frame gives those)."""
        train_geo = torch.is_grad_enabled() and any(p.requires_grad for p in self.LevelSetsNet.parameters())
        phi = self.geo_phi(x, z_geo)                                       # AUDIT 3: plain forward
        return phi if train_geo else phi.detach()

    def jump_step(self, x, rows, z_geo, z_field, phi, W_regions, W_grid, train_geo):
        """JUMP module. Band points get their front frame at t_n, the front speed from the input field,
        then advection to t_n+1 and the shock constraint there. 'free' runs it for the hooks but
        changes nothing. Returns (W_regions, jump, phi_next)."""
        cfg = self.shock_cfg
        reach = cfg['width'] + cfg['taper']
        d = phi.detach().argmax(dim=-1)                         # own zone

        # AUDIT 1 fix: j is the zone across the nearest front that is there (label check built in), chosen
        # by distance. The score rank says nothing about which zone is adjacent. Band points at t_n: a front
        # moves at most reach per step (advect), so dist <= 2 * reach can end within reach at t_n+1
        near, zones = front_distances(self.geo_phi, x, z_geo, d, cfg['side_offset']).topk(2, dim=-1, largest=False)
        j = zones[:, 0]
        idx = torch.nonzero(near[:, 0] <= 2 * reach).squeeze(1)
        z_band = z_geo[idx]
        frame = front_frame(lambda u: self.geo_phi(u, z_band), x[idx], d[idx], j[idx], train_geo)

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

        def fields_at(v, r):
            return self.heads_fn(self.FieldNet(MSFourier_Features(v, self.B_field), z_field[r]))

        W_regions, jump = constrain_shocks(W_regions, idx[keep], d_next[keep], j_next[keep],
                                           {key: val[keep] for key, val in frame_next.items()}, s_next[keep],
                                           (W_in_d[keep], W_in_j[keep]), fields_at, cfg,
                                           apply=self.jump_mode == 'rh')
        jump.update(flipped=flipped[keep], junction_frac=junction.float().mean())
        return W_regions, jump, phi_next

    def forward(self, x, z_geo, z_field, W_grid, group) -> dict:     # AUDIT fix: no blend / eps (hard only)

        '''
        INPUT REQUIREMENTS
        - x [N, D] normalized cell centres (space only), where W at t_n+1 is predicted
        - z_geo and z_field [N, .] per point, from the encoder of the input step
        - W_grid [B, C, N_grid] the input field at t_n (primitives); group [N] each point's row in W_grid

        OUTPUT DESCRIPTION (all at t_n+1 unless marked):
        - W : Field Variables at coordinate
        - W_regions : What each regions predicts for the Field Variables
        - labels : the region the point is assigned too
        - phi_hat : geometry's raw score phi for each region (advected);  phi_hat_now : the same at t_n
          (AUDIT 3: raw phi, not normalized; key names kept so evaluation/ still reads them)
        - weights : one-hot blend weights from the labels
        - jump : per band point hooks from the shock constraint (see constrain_shocks), None for 'none'
        '''

        N = x.shape[0]

        # for the BASELINE MODEL
        if self.geo_mode == 'none':
            z_cat = torch.cat([z_geo, z_field], dim=-1)

            W_regions = self.heads_fn(self.FieldNet(MSFourier_Features(x, self.B_field), z_cat))
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
        phi_now = self.geometry(x, z_geo)
        train_geo = phi_now.requires_grad

        #forward FIELD module, at t_n+1
        W_regions = self.heads_fn(self.FieldNet(MSFourier_Features(x, self.B_field), z_field))     # AUDIT 7

        #forward JUMP module: front speed, advection to t_n+1, pinned RH constraint behind shocks
        W_regions, jump, phi = self.jump_step(x, group, z_geo, z_field, phi_now,
                                              W_regions, W_grid, train_geo)

        #BLEND (assembly), hard, with the advected scores. AUDIT fix: the soft blend is gone; trained against a
        #~1 cell shock it made each zone field overshoot behind the front, and the hard-blend output kept it
        labels = phi.argmax(dim=-1)
        weights = nn.functional.one_hot(labels, phi.shape[-1]).to(W_regions)
        W = (weights.unsqueeze(-1) * W_regions).sum(dim=1)

        return {'W': W,
                'W_regions': W_regions,
                'labels': labels,
                'phi_hat': phi,
                'phi_hat_now': phi_now,
                'weights': weights,
                'jump': jump}
    