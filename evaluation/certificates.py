import torch


# ======================================================================================
# RH residual certificate (evaluation only). Moved from model/shock_interface.py
# ======================================================================================

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
