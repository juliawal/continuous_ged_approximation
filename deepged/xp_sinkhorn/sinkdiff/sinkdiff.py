import torch


def sinkhorn_d1d2(S, nb_iter=100, eps=1e-2):
    """
    Algorithm 1 in ICPR 2022

    S : similarity matrix 
    """
    n, m = S.shape
    ones_m = torch.ones(m, device=S.device)
    ones_n = torch.ones(n, device=S.device)
    c = ones_m
    converged = False
    i = 0
    while i <= nb_iter and not converged:
        rp = 1.0/(torch.matmul(S, c))
        rp[-1] = 1.0
        if i >= 1:
            norm_r = torch.linalg.norm(
                r/rp-ones_n, ord=float('inf'))
        r = rp
        cp = 1.0/(torch.matmul(S.T, r))
        cp[-1] = 1.0
        norm_c = torch.linalg.norm(
            c/cp-ones_m, ord=float('inf'))
        c = cp
        if i >= 1:
            converged = (norm_r <= eps) and (norm_c <= eps)
        i += 1
    # return assignement matrix
    return torch.diag(r)@S@torch.diag(c), i
