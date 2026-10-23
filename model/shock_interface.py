import torch


def mn_effective(Mn: torch.Tensor, mn_floor=0.5, beta: float = 50.0) -> torch.Tensor:
    """Smooth floor: Mn_e = m0 + softplus_beta(Mn - m0). mn_floor=None turns it off.
 
    Torch's softplus is exactly linear once beta*(Mn - m0) > 20, so with m0 = 0.5, beta = 50,
    Mn_e == Mn bit for bit for Mn > 0.9: every physical shock is untouched.
    """
    if mn_floor is None:
        return Mn
    return mn_floor + torch.nn.functional.softplus(Mn - mn_floor, beta=beta)
 
 
def Rankine_Hugoniot(W_minus: torch.Tensor, n: torch.Tensor, s: torch.Tensor,
           gamma: float = 1.4, mn_floor=0.5, beta: float = 50.0):
    """(W_minus [M, 3], n [M], s [M]) -> (W_plus [M, 3], Mn_raw [M]).
 
    Mn_raw is returned for the Lax certificate; the floored Mn_e is what the map uses.
    """
    rho_m, v_m, p_m = W_minus.unbind(-1)
    g = gamma
    c_m = torch.sqrt(g * p_m / rho_m)
    vn_m = v_m * n
    Mn = (vn_m - s) / c_m                           # raw
    M2 = mn_effective(Mn, mn_floor, beta) ** 2      # floored, used below
 
    rho_p = rho_m * (g + 1) * M2 / ((g - 1) * M2 + 2)
    p_p = p_m * (1 + 2 * g / (g + 1) * (M2 - 1))
    # Mass flux rho (v·n - s) is continuous: exact even when the floor is active.
    vn_p = s + (vn_m - s) * rho_m / rho_p
    v_p = v_m + (vn_p - vn_m) * n                   # only the normal part changes (tangential kept, for 2D)
    return torch.stack([rho_p, v_p, p_p], dim=-1), Mn
 
 
def prim_to_cons(W: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """(rho, v, p) -> (rho, m = rho v, E = p/(gamma-1) + rho v^2 / 2)."""
    rho, v, p = W.unbind(-1)
    return torch.stack([rho, rho * v, p / (gamma - 1) + 0.5 * rho * v * v], dim=-1)
 
 
def euler_flux(W: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """x-flux F(U) = (rho v, rho v^2 + p, v (E + p)), from primitives."""
    rho, v, p = W.unbind(-1)
    E = p / (gamma - 1) + 0.5 * rho * v * v
    return torch.stack([rho * v, rho * v * v + p, v * (E + p)], dim=-1)
 
 
def RH_residual(W_minus, W_plus, s, n, gamma: float = 1.4) -> torch.Tensor:
    """[M]: ||F(U+) - F(U-) - S (U+ - U-)|| / ||F(U-)||, with lab-frame speed S = s·n."""
    
    S = (s * n).unsqueeze(-1)
    dF = euler_flux(W_plus, gamma) - euler_flux(W_minus, gamma)
    dU = prim_to_cons(W_plus, gamma) - prim_to_cons(W_minus, gamma)
    num = torch.linalg.vector_norm(dF - S * dU, dim=-1)
    return num / torch.linalg.vector_norm(euler_flux(W_minus, gamma), dim=-1)
 
 
def lax_margin(Mn: torch.Tensor) -> torch.Tensor:
    """Mn_raw - 1: > 0 for an admissible (compressive) shock."""
    return Mn - 1
 
 
def char_speeds(W: torch.Tensor, gamma: float = 1.4) -> torch.Tensor:
    """[..., 3] lab-frame characteristic speeds (v - c, v, v + c)."""
    rho, v, p = W.unbind(-1)
    c = torch.sqrt(gamma * p / rho)
    return torch.stack([v - c, v, v + c], dim=-1)








def detect_shocks(W_a, W_b, n, s, gamma: float = 1.4, mn_floor=0.5, beta: float = 50.0,
                  tau_p: float = 0.05) -> dict:
    """Classify each front as a shock or not, and give the RH state behind each shock.
 
    W_a, W_b [F, 3]: the fields of the zones left (a) and right (b) of each front, read at the
    front point. n [F] points from a to b; s [F] is the normal speed along n, in field units
    (both from find_fronts).
 
    Upstream is the low-pressure side. n_ud points upstream -> downstream, and s_ud is the normal
    speed along n_ud. A front is a shock when p_down > (1 + tau_p) p_up and the flow compresses,
    (v_down - v_up) * n_ud < 0. That is a discrete decision, with no gradient.
    Sod check: shock between zone 3 (a, left) and 4 (b, right): upstream = b, n_ud = -1,
    v_down - v_up = u* > 0, so the compression test holds.
 
    Returns [F] tensors: is_shock (bool), up_is_a (bool), n_ud, s_ud, W_up [F, 3],
    W_RH [F, 3] and Mn_raw, Mn_eff [F]. The last three are NaN where is_shock is False:
    the RH map is only evaluated at shocks.
    """
    up_is_a = W_a[:, 2] <= W_b[:, 2]
    W_up = torch.where(up_is_a[:, None], W_a, W_b)
    W_dn = torch.where(up_is_a[:, None], W_b, W_a)
    n_ud = torch.where(up_is_a, n, -n)
    s_ud = torch.where(up_is_a, s, -s)
 
    with torch.no_grad():
        is_shock = (W_dn[:, 2] > (1 + tau_p) * W_up[:, 2]) & ((W_dn[:, 1] - W_up[:, 1]) * n_ud < 0)
 
    F = W_a.shape[0]
    idx = torch.nonzero(is_shock).squeeze(1)
    W_RH_s, Mn_s = Rankine_Hugoniot(W_up[idx], n_ud[idx], s_ud[idx], gamma, mn_floor, beta)
    nan = float("nan")
    W_RH = W_up.new_full((F, 3), nan).index_put((idx,), W_RH_s)
    Mn_raw = W_up.new_full((F,), nan).index_put((idx,), Mn_s)
    Mn_eff = W_up.new_full((F,), nan).index_put((idx,), mn_effective(Mn_s, mn_floor, beta))
 
    return {"is_shock": is_shock, "up_is_a": up_is_a, "n_ud": n_ud, "s_ud": s_ud,
            "W_up": W_up, "W_RH": W_RH, "Mn_raw": Mn_raw, "Mn_eff": Mn_eff}



