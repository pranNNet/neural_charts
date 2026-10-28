import torch
import torch.nn as nn

from model.transforms import build_B, MSFourier_Features
from model.geometry import LevelSetsNet, zone_scores, front_frame
from model.region import FieldNet, positive_clamp
from model.shock_interface import constrain_shocks

# ======================================================================================
# NeuralChart: GEOMETRY -> FIELD -> JUMP -> BLEND
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
                                     N_dist = field_cfg['fieldnet']['N_dist'],
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers'],
                                     N_out = field_cfg['fieldnet']['field_output']
                                     )
            # relevant parameters for combinations
            self.norm_delta = geo_cfg['norm_delta']

            # shock constraint (JUMP module)
            jump_cfg, front_cfg = config['physics']['jump'], config['physics']['fronts']
            self.shock_cfg = {
                'gamma': config['physics']['constants']['gamma'],
                'k': front_cfg['k'],                                 # x~/t~ speed -> field units
                'tau_p': front_cfg['tau_p'],                         # pressure-rise threshold
                'Mn_gate': jump_cfg.get('Mn_gate', 1.05),            # Lax gate
                'side_offset': front_cfg['side_offset'],             # x~, plateau read distance
                't_min': front_cfg['t_min'],                         # t~, no shocks before
                'width': jump_cfg['width'],                          # x~, full-strength constraint
                'taper': jump_cfg['taper'],                          # x~, cosine taper after width
            }
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
                                     N_dist = 0,
                                     hidden_layers = field_cfg['fieldnet']['hidden_layers'],
                                     N_out = field_cfg['fieldnet']['field_output']
                                     )



    def geo_phi(self, u, z):
        """u [P, D+1] normalized (space..., t~), z [P, z_geo] -> phi [P, K]. Same call as in forward."""
        return self.LevelSetsNet(MSFourier_Features(u, self.B_geo), z)


    def heads_fn(self, h):
        """S.6, switched by `heads`. linear: rho = h1, p = h3 (positivity ablation, §6)."""
        return positive_clamp(h) if self.head == "softplus" else h


    def jump_step(self, x, z_geo, z_field, phi_hat, W_regions, train_geo):
        """JUMP module. Band points (within width + taper of a front) get their front frame from the
        geometry, then the shock constraint. 'free' runs it for the hooks but changes nothing.
        Returns (W_regions, jump)."""
        reach = self.shock_cfg['width'] + self.shock_cfg['taper']
        top, zones = phi_hat.detach().topk(2, dim=-1)
        d, j = zones.unbind(-1)                                 # own zone, zone across the nearest front

        # pre-filter on psi = phi_hat_d - phi_hat_j, about 2 * dist where phi_hat is distance-like
        idx = torch.nonzero(top[:, 0] - top[:, 1] <= 4 * reach).squeeze(1)       # [GUESS] 2x margin
        z_band = z_geo[idx]
        frame = front_frame(lambda u: self.geo_phi(u, z_band), x[idx], d[idx], j[idx],
                            self.norm_delta, train_geo)
        near = frame['dist'].detach() <= reach
        idx, frame = idx[near], {key: v[near] for key, v in frame.items()}

        def fields_at(u, rows):
            return self.heads_fn(self.FieldNet(MSFourier_Features(u, self.B_field), z_field[rows]))

        def labels_at(u, rows):
            return zone_scores(lambda v: self.geo_phi(v, z_geo[rows].detach()), u.detach(),
                               self.norm_delta, False).argmax(dim=-1)

        return constrain_shocks(W_regions, x, idx, d, j, frame, fields_at, labels_at,
                                self.shock_cfg, apply=self.jump_mode == 'rh')


    def forward(self, x, z_geo, z_field,
                blend = 'hard',
                eps: float = None,
                group = None) -> dict:      # group: unused, kept so current callers still run

        '''
        INPUT REQUIREMENTS
        - x must be normalized
        - z_geo and z_field must be precomputed vectors from the encoder

        OUTPUT DESCRIPTION:
        - W : Field Variables at coordinate
        - W_regions : What each regions predicts for the Field Variables
        - labels : the region the point is assigned too
        - phi_hat : geometry's score for each region
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
                    'weights': x.new_ones(N, 1),
                    'jump': None}


        # for the LEVELSET MODEL

        #forward GEOMETRY module
        grad_on = torch.is_grad_enabled()

        # graph needed for training
        train_geo = grad_on and any(p.requires_grad for p in self.LevelSetsNet.parameters())

        phi_hat = zone_scores(lambda u: self.geo_phi(u, z_geo), x, self.norm_delta, train_geo)

        # evaluation
        if not train_geo:
            phi_hat = phi_hat.detach()


        #forward FIELD module
        dist = x.new_zeros(N, self.FieldNet.N_dist)
        W_regions = self.heads_fn(self.FieldNet(MSFourier_Features(x, self.B_field), z_field, dist))


        #forward JUMP module: pinned RH constraint behind shocks
        W_regions, jump = self.jump_step(x, z_geo, z_field, phi_hat, W_regions, train_geo)


        #BLEND (assembly)
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
                'weights': weights,
                'jump': jump}
