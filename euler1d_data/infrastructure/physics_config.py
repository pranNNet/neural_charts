'''
Solver Configuration Code
- Creates physical 1D Euler Problem [IC, BC, gamma]
    - "Randomly" instantiates a problem
        - Using set seed though to help with reproducability
    - This is specificially to define and generate various physical problem definition
        - For numerical problem setup, go to numerical_config.py
'''
from dataclasses import dataclass, field
import numpy as np


from infrastructure.results import ExactRiemannSolver


split_generator = np.random.default_rng(seed = 12345)
vel_generator = np.random.default_rng(seed = 42)
dens_generator = np.random.default_rng(seed = 0)
pres_generator = np.random.default_rng(seed = 67)


# @dataclass()
# class PhysicsConfig:
#     '''
#     Configuration for a 1D Euler Problem
#     - Generatees a 1D Euler Problem with random initial conditions
#         - Uniform sampling with fixed seeds for generator
#         - ranges arbitrarily chosen
#             - values clamped to positivity
#             - Bounds of generation ranges arbitraily chosen
#         - Does need N_cells from user
#     - Does contain nescessary input parameters to exact_Reimann_solver function
    
#     ***ATTRIBUTES***
#     BC
#     N_cells
#     gamma
#     Q_L, Q_R
#     x_split
#     IC
#     domain_split_percent
#     '''

#     #boundary conditions
#     BC: str = field(default = 'Zero-Gradient') #Zero-Gradient, Reflective, Periodical


#     # other parameters
#     N_cells: int = field(default = 500) #N cells in domain
#     gamma: float = field(default = 1.4) #ratio of specific heats

#     use_Sod: bool = field(default = False) #use the Sod shock tube setup



#     def __post_init__(self):
#         def get_cons(rho, u, P, gamma = self.gamma):
#             mom = rho * u
#             E = (P / (gamma - 1.0)) + 0.5 * rho * u**2
#             return np.array([rho, mom, E])

#         rho_L = dens_generator.uniform(0.1, 2.5)
#         rho_R = dens_generator.uniform(0.1, 2.5)

#         #normal chosen to center around u = 0, like in the sod shock tube setup
#         u_L = 0 #vel_generator.normal(0., 5.) 
#         u_R = 0 #vel_generator.normal(0., 5.) 

#         P_L = pres_generator.uniform(10000., 125000)
#         P_R = pres_generator.uniform(10000., 125000.)


#         # Hardcode use Sod Shock Tube
#         if self.use_Sod:
#             rho_L, u_L, P_L = 1.0, 0, 100000.0
#             rho_R, u_R, P_R = 0.125, 0, 10000.0

                    
#         # conservative 
#         self.Q_L = get_cons(rho_L, u_L, P_L, self.gamma)
#         self.Q_R = get_cons(rho_R, u_R, P_R, self.gamma)


#         #Init Condition of Conservative State Matrix
#         self.IC = np.zeros((3, self.N_cells)) 
#         self.x_split = split_generator.integers(int(self.N_cells*0.2), int(self.N_cells*0.8), endpoint = True)
#         if self.use_Sod:
#             self.x_split = int(self.N_cells*0.5)
#         self.IC[:, self.x_split] = self.Q_L
#         for i in range(self.N_cells):
#             self.IC[:, i] = self.Q_L if i < self.x_split else self.Q_R

#         # domain split location in terms of %domain from the left
#         self.domain_split_percent = self.x_split/self.N_cells


# NEW: fixed left state, u = 0 on both sides. P_L is deliberately 1e3 (not Sod's 1e5): at fixed dx, dt and t_max the
# CFL number (S*dt/dx) and wave travel distance (S*t_max) both scale with sound speed ~ sqrt(P/rho), so a lower
# reference pressure is what lets you keep dt = 1e-6, t_max = 1e-3 with a wide ratio box.
RHO_L_REF, P_L_REF = 1.0, 1.0e5
 
 
# NEW ------------------------------------------------------------------------------------------
def _log_uniform(u, lo, hi):
    """u in [0,1) -> value in [lo, hi], uniform in log space."""
    return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
 
 
def riemann_speeds(sol):
    """From an ExactRiemannSolver: (slowest jump-wave speed, fastest jump-wave speed, max signal speed).
    Jump waves = shocks + contact (+ the diaphragm at speed 0). Signal speed = max(|u| + a) over every
    constant state and shock speed, i.e. what the CFL condition cares about."""
    g, Ps = sol.gamma, sol.P_star
    jumps = [0.0, sol.u_star]
    sig = [sol.a_L, sol.a_R,                                              # u = 0 initial states
           abs(sol.u_star) + np.sqrt(g * Ps / sol.rho_star_L),            # star states
           abs(sol.u_star) + np.sqrt(g * Ps / sol.rho_star_R)]
    for Pk, uk, ak, sign in ((sol.P_L, sol.u_L, sol.a_L, -1.0), (sol.P_R, sol.u_R, sol.a_R, +1.0)):
        if Ps > Pk:                                                       # shock speed, same formula as sample()
            S = uk + sign * ak * np.sqrt((g + 1) / (2 * g) * (Ps / Pk) + (g - 1) / (2 * g))
            jumps.append(S); sig.append(abs(S))
    return min(jumps), max(jumps), max(sig)
 
 
def check_sampling_ranges(rho_ratio_range=(0.1, 10.0), p_ratio_range=(0.1, 10.0),
                          x_split_range=(0.3, 0.7), domain_size=1.0, N_cells=1000,
                          wave_margin=0.05, cfl_max=0.5, n_grid=25):
    """Run ONCE before generating data. Scans the ratio box (log grid, corners included) and returns the
    LARGEST t_max and dt that keep every IC valid:
      t_max_ok : shock/contact stay >= wave_margin*domain away from both walls for every x0 in x_split_range
      dt_ok    : CFL = S_signal*dt/dx <= cfl_max everywhere in the box
    Use t_max <= t_max_ok and dt <= dt_ok (or shrink the ranges)."""
    rr = np.exp(np.linspace(np.log(rho_ratio_range[0]), np.log(rho_ratio_range[1]), n_grid))
    pr = np.exp(np.linspace(np.log(p_ratio_range[0]), np.log(p_ratio_range[1]), n_grid))
    sp = np.array([riemann_speeds(ExactRiemannSolver((RHO_L_REF, 0.0, P_L_REF),
                                                     (RHO_L_REF * a, 0.0, P_L_REF * b)))
                   for a in rr for b in pr])
    jmin, jmax, smax = sp[:, 0].min(), sp[:, 1].max(), sp[:, 2].max()
    room_left = (x_split_range[0] - wave_margin) * domain_size
    room_right = (1 - wave_margin - x_split_range[1]) * domain_size
    out = {"slowest_jump": jmin, "fastest_jump": jmax, "max_signal_speed": smax,
           "t_max_ok": min(room_left / max(-jmin, 1e-12), room_right / max(jmax, 1e-12)),
           "dt_ok": cfl_max * (domain_size / N_cells) / smax}
    print({k: float(f"{v:.4g}") for k, v in out.items()})
    return out
 
 
def make_lhs_design(n, seed, optimization="random-cd"):
    """(n, 3) Latin hypercube in [0,1)^3 -> columns (u_rho, u_P, u_x). Use a DIFFERENT seed for train,
    validate and test. Row i goes to PhysicsConfig(unit_sample=tuple(design[i]))."""
    from scipy.stats import qmc
    return qmc.LatinHypercube(d=3, seed=seed, optimization=optimization).random(n)
# ----------------------------------------------------------------------------------------------
 
 
@dataclass()
class PhysicsConfig:
    '''
    Configuration for a 1D Euler Problem (Sod-type Riemann problem, u = 0, fixed left state)
    - Right state set by log-uniformly sampled RATIOS rho_R/rho_L and P_R/P_L
    - Diaphragm location sampled uniformly in x_split_range
    - Optional unit_sample = (u_rho, u_P, u_x) in [0,1)^3 (e.g. a Latin-hypercube row) replaces the generators
    '''
 
    #boundary conditions
    BC: str = field(default = 'Zero-Gradient') #Zero-Gradient, Reflective, Periodical
 
    # other parameters
    N_cells: int = field(default = 500) #N cells in domain
    gamma: float = field(default = 1.4) #ratio of specific heats
 
    use_Sod: bool = field(default = False) #use the Sod shock tube setup
 
    # NEW: sampling and validity parameters
    rho_ratio_range: tuple = (0.1, 10.0)   # rho_R / rho_L, log-uniform
    p_ratio_range: tuple = (0.1, 10.0)     # P_R / P_L, log-uniform (ratio = 1 is the shock/rarefaction switch)
    x_split_range: tuple = (0.2, 0.8)      # diaphragm as fraction of domain (your original range)
    unit_sample: tuple = None              # (u_rho, u_P, u_x) in [0,1)^3, or None -> use the generators
    domain_size: float = 1.0
    t_max: float = 2.0e-4                  # must match random_Run; verify with check_sampling_ranges()
    dt: float = 1.0e-7                     # must match random_Run; verify with check_sampling_ranges()
    wave_margin: float = 0.05              # shock/contact must stay this far (domain fraction) from the walls
    cfl_max: float = 0.5
 
    def __post_init__(self):
        def get_cons(rho, u, P, gamma = self.gamma):
            mom = rho * u
            E = (P / (gamma - 1.0)) + 0.5 * rho * u**2
            return np.array([rho, mom, E])
 
        # CHANGED: unit-cube sample -> ratios and x0 (instead of uniform values for rho_L, rho_R, P_L, P_R)
        # if self.unit_sample is not None:
        #     u_rho, u_P, u_x = self.unit_sample
        # else:
        #     u_rho, u_P, u_x = dens_generator.uniform(), pres_generator.uniform(), split_generator.uniform()
        if self.unit_sample is not None:
            _, u_P, _ = self.unit_sample
            u_x = 0.5
        else:
            u_P = pres_generator.uniform()

            u_rho = 0.5
            u_x = 0.5
 
        self.rho_ratio = 0.125 #_log_uniform(u_rho, *self.rho_ratio_range)
        self.p_ratio = _log_uniform(u_P, *self.p_ratio_range)
        self.x_split = int(round(self.N_cells * (self.x_split_range[0]
                                                 + u_x * (self.x_split_range[1] - self.x_split_range[0]))))
 
        rho_L, u_L, P_L = RHO_L_REF, 0.0, P_L_REF
        rho_R, u_R, P_R = rho_L * self.rho_ratio, 0.0, P_L * self.p_ratio
 
        # Hardcode use Sod Shock Tube
        if self.use_Sod:
            rho_L, u_L, P_L = 1.0, 0, P_L_REF            # CHANGED: Sod is self-similar, scaled to the reference pressure
            rho_R, u_R, P_R = 0.125, 0, 0.1 * P_L_REF
            self.rho_ratio, self.p_ratio = rho_R / rho_L, P_R / P_L
            self.x_split = int(self.N_cells*0.5)
 
        # conservative 
        self.Q_L = get_cons(rho_L, u_L, P_L, self.gamma)
        self.Q_R = get_cons(rho_R, u_R, P_R, self.gamma)
 
        #Init Condition of Conservative State Matrix
        self.IC = np.zeros((3, self.N_cells)) 
        self.IC[:, self.x_split] = self.Q_L
        for i in range(self.N_cells):
            self.IC[:, i] = self.Q_L if i < self.x_split else self.Q_R
 
        # domain split location in terms of %domain from the left
        self.domain_split_percent = self.x_split/self.N_cells
 
        # NEW: per-IC safety checks with the exact solver
        sol = ExactRiemannSolver((rho_L, u_L, P_L), (rho_R, u_R, P_R),
                                 x_interface=self.domain_split_percent * self.domain_size, gamma=self.gamma)
        jmin, jmax, smax = riemann_speeds(sol)
        x0 = self.domain_split_percent * self.domain_size
        lo, hi = self.wave_margin * self.domain_size, (1 - self.wave_margin) * self.domain_size
        self.cfl = smax * self.dt / (self.domain_size / self.N_cells)
        self.waves_contained = (x0 + jmin * self.t_max >= lo) and (x0 + jmax * self.t_max <= hi)
        if self.cfl > self.cfl_max:
            raise ValueError(f"CFL {self.cfl:.2f} > {self.cfl_max} (ratios {self.rho_ratio:.3g}, {self.p_ratio:.3g}): "
                             f"reduce dt or the ratio ranges (see check_sampling_ranges)")
        if not self.waves_contained:
            raise ValueError(f"waves leave the domain before t_max (x0={x0:.2f}, speeds {jmin:.0f}..{jmax:.0f}): "
                             f"reduce t_max or x_split_range (see check_sampling_ranges)")
 
 


