import yaml
from functools import partial

import torch
import torch.nn as nn

from model.transforms import build_B, make_B, MSFourier_Features
from model.geometry import LevelSetsNet
from model.region import FieldNet, positive_clamp
from model.shock_interface import Rankine_Hugoniot, detect_shocks
from model.helper import phi_and_slope, search_points, find_fronts, split_fronts

# ======================================================================================
# FieldNet
# ======================================================================================

class NeuralChart1DST(nn.Module):

    def __init__(self, config: dict, 
                 N_input = 2):
        super().__init__()

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
                                 fourier=self.fourier
                                 )

            self.B_field = build_B(bands = field_cfg['fourierfeatures']['bands'],
                                   seed = field_cfg['fourierfeatures']['seed'],
                                   fourier=self.fourier
                                   )


            self.LevelSetsNet = LevelSetsNet(N_in = N_input + N_input*self.B_geo.shape[1],
                                            z_geo = config['architecture']['latent']['z_geo'],
                                            N_out = geo_cfg['geometrynet']['phi_output'],
                                            hidden_layers = geo_cfg['geometrynet']['hidden_layers']
                                            )

            self.FieldNet = FieldNet(N_in = N_input + N_input*self.B_field.shape[1],
                                     N_regions = field_cfg['fieldnet']['regions'],
                                     N_z_field = config['architecture']['latent']['z_field'],
                                     N_e_field = field_cfg['fieldnet']['N_e'],
                                     N_dist = field_cfg['fieldnet']['N_dist'],
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers']
                                     )
            # relevant parameters for combinations
            self.norm_delta = geo_cfg['norm_delta']
            self.N_search = config['physics']['fronts']['N_search'] # numb. of sample points for shock search
            self.search_chunk = config['physics']['fronts']['search_chunk']
            self.k = config['physics']['fronts']['k'] #convert normalzed shock speed to field units

            self.gamma = config['physics']['constants']['gamma']
            self.mn_floor = config['physics']['jump']['Mn_floor']
            self.beta = config['physics']['jump']['beta']
            self.tau_p = config['physics']['fronts']['tau_p']
            # [FIX 1-3] new keys, with defaults so old yamls still load
            self.mn_gate = config['physics']['jump'].get('Mn_gate', 1.05)                  # Lax gate on is_shock
            self.side_offset = config['physics']['fronts'].get('side_offset', 4 * 2 / 512)  # x~, 4 cells
            self.t_min = config['physics']['fronts'].get('t_min', 0.02 / 0.16)             # t~, as jump_flags
 

        else:
            # Baseline comparison (no Level Set Functionality)


            #bands for multi-scale fourier features
            self.B_geo, self.LevelSetsNet = None, None

            self.B_field = build_B(bands = field_cfg['fourierfeatures']['bands'],
                                   seed = field_cfg['fourierfeatures']['seed'],
                                   fourier=self.fourier
                                   )


            self.FieldNet = FieldNet(N_in = N_input + N_input*self.B_field.shape[1],
                                     N_regions = 1,
                                     N_z_field = config['architecture']['latent']['z_field'] + config['architecture']['latent']['z_geo'],
                                     N_e_field = 1,
                                     N_dist = 0,
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers']
                                     )


        self.RH_map = partial(Rankine_Hugoniot,
                              gamma = config['physics']['constants']['gamma'],
                              beta = config['physics']['jump']['beta'],
                              mn_floor = config['physics']['jump']['Mn_floor']
                              )


    def geo_phi(self, u, z):
        """u [P, 2] normalized (x~, t~), z [P, z_geo] -> phi [P, K]. Same call as in forward."""
        return self.LevelSetsNet(MSFourier_Features(u, self.B_geo), z)


    def heads_fn(self, h):
        """S.6, switched by `heads`. linear: rho = h1, p = h3 (positivity ablation, §6)."""
        return positive_clamp(h) if self.head == "softplus" else h


    # def fronts_for_group(self, x, z_geo, z_field, in_g, train_geo) -> dict:
    #     """Fronts of one (case, time) group, and which of them are shocks.
 
    #     Returns find_fronts' dict (x, dxdt, n, s, a, b) plus detect_shocks' keys
    #     (is_shock, up_is_a, n_ud, s_ud, W_up, W_RH, Mn_raw, Mn_eff), all of length F.
    #     """
    #     # Every point of a group shares one case and one time: read them off the first point.
    #     first = torch.nonzero(in_g)[0, 0]
    #     t_g = x[first, 1].detach()
    #     z_geo_g, z_field_g = z_geo[first], z_field[first]
 
    #     # 1. Where the regions change along x~ at this time.
    #     x_search = search_points(self.N_search).to(x)
    #     fr = find_fronts(self.geo_phi, z_geo_g, t_g, x_search,
    #                      self.k, self.norm_delta, train_geo)
 
    #     # 2. The state each side's region predicts at the front point.
    #     F = len(fr['x'])
    #     u = torch.stack([fr['x'], t_g.expand(F)], dim=-1)                       # [F, 2]
    #     W_front = self.heads_fn(self.FieldNet(MSFourier_Features(u, self.B_field),
    #                                           z_field_g.expand(F, -1)))          # [F, K, 3]
 
    #     rows = torch.arange(F, device=x.device)
    #     W_a = W_front[rows, fr['a']]                                            # [F, 3] left side
    #     W_b = W_front[rows, fr['b']]                                            # [F, 3] right side
 
    #     # 3. Shocks: pressure rise with compression. Gives the RH state behind each, and Mn.
    #     fr.update(detect_shocks(W_a, W_b, fr['n'], fr['s'], self.gamma,
    #                             self.mn_floor, self.beta, self.tau_p))
    #     return fr
 
    # def overwrite_shocked_zones(self, W_regions, fr, in_g, g):
    #     """In place: the region behind each shock takes that shock's RH state at every point
    #     of the group. The RH state is constant in x, which is exact for Sod's star-right region.
    #     """
    #     done = set()
    #     for f in torch.nonzero(fr['is_shock']).squeeze(1).tolist():
 
    #         # downstream = the high-pressure side of the shock
    #         down = int(fr['b'][f] if fr['up_is_a'][f] else fr['a'][f])
 
    #         # Open case (D78): a region behind two shocks keeps the first one.
    #         if down in done:
    #             print(f"warning: region {down} is behind two shocks in group {g}; keeping the first")
    #             continue
    #         done.add(down)
 
    #         W_regions[in_g, down] = fr['W_RH'][f]

    def fronts_all_groups(self, x, z_geo, z_field, group, train_geo):
        """Fronts of every (case, time) group in one pass, and which of them are shocks.
 
        Returns (fr, uniq, inv). fr is find_fronts' flat dict (g, x, dxdt, n, s, a, b) plus
        detect_shocks' keys (is_shock, up_is_a, n_ud, s_ud, W_up, W_RH, Mn_raw, Mn_eff), all of
        length F = total fronts. fr['g'] indexes uniq (the sorted group ids); inv [N] maps each
        point to its row of uniq.
        """
        # Every point of a group shares one case and one time: read them off its first point.
        uniq, inv = torch.unique(group, return_inverse=True)
        N = len(group)
        ar = torch.arange(N, device=x.device)
        first = torch.full((len(uniq),), N, device=x.device, dtype=torch.long).scatter_reduce(
            0, inv, ar, reduce='amin')                                          # [G]
        t_g = x[first, 1].detach()
        z_geo_g, z_field_g = z_geo[first], z_field[first]
 
        # 1. Where the regions change along x~, all groups' search grids stacked.
        x_search = search_points(self.N_search).to(x)
        fr = find_fronts(self.geo_phi, z_geo_g, t_g, x_search,
                         self.k, self.norm_delta, train_geo, self.search_chunk)
 
        # 2. [FIX 2] The state each side's region predicts side_offset away from the front
        #    (zone a at x - d, zone b at x + d), so the plateau is read and not the smeared ramp.
        #    One FieldNet call on both sides stacked.
        F = len(fr['x'])
        # u = torch.stack([fr['x'], t_g[fr['g']]], dim=-1)                        # [F, 2]
        # W_front = self.heads_fn(self.FieldNet(MSFourier_Features(u, self.B_field),
        #                                       z_field_g[fr['g']]))               # [F, K, 3]
        d = self.side_offset
        t_f = t_g[fr['g']]
        u = torch.stack([torch.cat([(fr['x'] - d).clamp(-1, 1), (fr['x'] + d).clamp(-1, 1)]),
                         t_f.repeat(2)], dim=-1)                                   # [2F, 2]
        W_side = self.heads_fn(self.FieldNet(MSFourier_Features(u, self.B_field),
                                             z_field_g[fr['g']].repeat(2, 1)))     # [2F, K, 3]
 
        rows = torch.arange(F, device=x.device)
        # W_a = W_front[rows, fr['a']]                                            # [F, 3] left side
        # W_b = W_front[rows, fr['b']]                                            # [F, 3] right side
 
        # # 3. Shocks: pressure rise with compression. Gives the RH state behind each, and Mn.
        W_a = W_side[:F][rows, fr['a']]                                         # [F, 3] left side, at x - d
        W_b = W_side[F:][rows, fr['b']]                                         # [F, 3] right side, at x + d
        fr.update(detect_shocks(W_a, W_b, fr['n'], fr['s'], self.gamma,
                                self.mn_floor, self.beta, self.tau_p, self.mn_gate))

        # 4. [FIX 3] No shocks before t_min: the geometry is never trained there (jump_flags = -1).
        fr['is_shock'] = fr['is_shock'] & (t_f >= self.t_min)
        return fr, uniq, inv
 
 
    def overwrite_shocked_zones(self, W_regions, fr, inv, uniq):
        """The region behind each shock takes that shock's RH state at every point of its group.
        The RH state is constant in x, which is exact for Sod's star-right region.
        Returns a new W_regions (out of place, so no clone is needed); gradients reach W_RH as before.
        """
        sh = torch.nonzero(fr['is_shock']).squeeze(1)
        # downstream = the high-pressure side of the shock
        down = torch.where(fr['up_is_a'][sh], fr['b'][sh], fr['a'][sh])
 
        # # Open case (D78): a region behind two shocks keeps the first one (fronts run left to right).
        # keep, done = [], set()
        # for f, gj, d in zip(sh.tolist(), fr['g'][sh].tolist(), down.tolist()):   # one sync, few shocks
        #     if (gj, d) in done:
        #         print(f"warning: region {d} is behind two shocks in group {uniq[gj].item()}; keeping the first")
        #         continue
        #     done.add((gj, d))
        #     keep.append(f)

        # [FIX 4] Open case (D78): a region behind two or more shocks is ambiguous, so it is not
        # overwritten at all (it keeps its free field). Counted in self.n_conflicts, not printed.
        pairs = list(zip(sh.tolist(), fr['g'][sh].tolist(), down.tolist()))     # one sync, few shocks
        count = {}
        for _, gj, d in pairs:
            count[(gj, d)] = count.get((gj, d), 0) + 1
        keep = [f for f, gj, d in pairs if count[(gj, d)] == 1]
        self.n_conflicts = sum(1 for c in count.values() if c > 1)
        if not keep:
            return W_regions
 
        keep = torch.tensor(keep, device=W_regions.device)
        gk = fr['g'][keep]
        dk = torch.where(fr['up_is_a'][keep], fr['b'][keep], fr['a'][keep])
        G, K = len(uniq), W_regions.shape[1]
 
        # [G, K] table of the RH state per (group, region), and where it applies
        table = W_regions.new_zeros(G, K, 3).index_put((gk, dk), fr['W_RH'][keep])
        hit = torch.zeros(G, K, dtype=torch.bool, device=W_regions.device).index_put(
            (gk, dk), torch.tensor(True, device=W_regions.device))
        return torch.where(hit[inv].unsqueeze(-1), table[inv], W_regions)
 

    def forward(self, x, z_geo, z_field,
                blend = 'hard',
                eps: float = None,
                group = None) -> dict:

        '''
        INPUT REQUIREMENTS
        - x must be normalized
        - z_geo and z_field must be precomputed vectors from the encoder

        OUTPUT DESCRIPTION:
        - W : Field Variables at coordinate
        - W_regions : What each regions predicts for the Field Variables
        - labels : the region the point is assigned too
        - phi : geometry's score for each region
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
                    # 'dphi_dx': None,
                    'phi_hat': None,
                    'weights': x.new_ones(N, 1),
                    'fronts': None}


        # for the LEVELSET MODEL

        #forward GEOMETRY module
        grad_on = torch.is_grad_enabled()

        # graph needed for training
        train_geo = grad_on and any(p.requires_grad for p in self.LevelSetsNet.parameters())

        phi, dphi_dx = phi_and_slope(
            phi_fn = lambda u: self.LevelSetsNet(MSFourier_Features(u, self.B_geo), z_geo),
            x = x,
            create_graph = train_geo
        )

        # evaluation
        if not train_geo:
            phi, dphi_dx = phi.detach(), dphi_dx.detach()

        phi_hat = phi/torch.sqrt(dphi_dx**2 + self.norm_delta**2)


        #forward FIELD module
        dist = x.new_zeros(N, self.FieldNet.N_dist)
        W_regions = self.heads_fn(self.FieldNet(MSFourier_Features(x, self.B_field), z_field, dist))


        # # JUMP: fronts and RH overwrite, one (case, time) group at a time
        # fronts = None
        # if group is not None:
        #     fronts = []
        #     W_regions = W_regions.clone()       # zones behind shocks are written into below
 
        #     for g in torch.unique(group).tolist():
        #         in_g = group == g               # [N] bool: the points of this group
 
        #         fr = self.fronts_for_group(x, z_geo, z_field, in_g, train_geo)
 
        #         # 'free' still finds fronts and shocks (for the certificates) but overwrites nothing
        #         if self.jump_mode == 'rh':
        #             self.overwrite_shocked_zones(W_regions, fr, in_g, g)
 
        #         fronts.append(fr)

        # ---- JUMP: fronts of all (case, time) groups at once, then the RH overwrite ----  [CLAUDE ADDED]
        fronts = None
        if group is not None:
            fr, uniq, inv = self.fronts_all_groups(x, z_geo, z_field, group, train_geo)
 
            # 'free' still finds fronts and shocks (for the certificates) but overwrites nothing
            if self.jump_mode == 'rh':
                W_regions = self.overwrite_shocked_zones(W_regions, fr, inv, uniq)
 
            fronts = split_fronts(fr, len(uniq))     # one dict per group, in sorted group order



        #ASSEMBLY
        labels = phi_hat.argmax(dim=-1)
        if blend == 'hard':
            weights = nn.functional.one_hot(labels, phi_hat.shape[-1]).to(phi_hat)
            W = torch.take_along_dim(W_regions, labels[:, None, None], dim=1).squeeze(1)

        else:
            weights = torch.softmax(phi_hat/eps, dim=-1)
            W = (weights.unsqueeze(-1) * W_regions).sum(dim=1)


        return {'W': W,
                'W_regions': W_regions,
                'labels': labels,
                'phi_hat': phi_hat,
                # 'dphi_dx': dphi_dx,
                'weights': weights, 
                'fronts': fronts}





