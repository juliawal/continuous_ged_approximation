import torch
import sys


def sinkhorn_diff(similarity_matrix, nb_iter):
    '''
    LE sinkhorn dérivable
    '''
    ones_m = torch.ones(
        similarity_matrix.shape[1], device=similarity_matrix.device)
    c = ones_m
    converged = False
    i = 0
    while i <= nb_iter and not converged:
        rp = 1.0/(similarity_matrix@c)
        rp[-1] = 1.0
        if i >= 1:
            norm_r = torch.linalg.norm(
                r/rp-torch.ones_like(r/rp), ord=float('inf'))
        r = rp
        cp = 1.0/(similarity_matrix.T@r)
        cp[-1] = 1.0
        norm_c = torch.linalg.norm(
            c/cp-torch.ones_like(c/cp), ord=float('inf'))
        c = cp
        if i >= 1:
            converged = (norm_r <= 1e-2) and (norm_c <= 1e-2)
        i += 1
    return torch.diag(r)@similarity_matrix@torch.diag(c)


def simplify_matrix(S):
    '''
    cf papier
    '''
    n = S.shape[0]-1
    m = S.shape[1]-1
    ones_n = torch.ones((n, 1), device=S.device)
    ones_m = torch.ones((1, m), device=S.device)
    C = ones_n@(S[-1, 0:m].reshape(1, m))
    # entry (i,j) equals to S[-1,j]+S[i,-1]
    C += S[0:n, -1].reshape(n, 1)@ones_m
    S2 = torch.empty((n+1, m+1), device=S.device)
    S2[-1, :] = S[-1, :]
    S2[:, -1] = S[:, -1]
    # If S[i,j]<S[-1,j]+S[i,-1] then S[i,j]=10^{-4}
    S2[0:n, 0:m] = torch.where(
        S[0:n, 0:m] < C, 1e-4*torch.ones((n, m), device=S.device), S[0:n, 0:m])

    return S2


def from_cost_to_similarity(C):
    '''
    Fonction décrite dans le papier. Ne semble pas permettre d'apprendre
    '''
    M = torch.max(C)+1.0
    S = 2*M*torch.ones((C.shape[0], C.shape[1]), device=C.device)
    S[-1, :] = M
    S[:, -1] = M
    return simplify_matrix(S-C)


def from_cost_to_similarity_exp(C):
    '''
    Transforme une matrice de couts en une matrice de similarité.
    Cette version semble marcher, au contraire de la précédente.
    '''
    n, m = C.shape
    ones_m = torch.ones((1, m))
    ones_n = torch.ones((n, 1))
    minL, _ = C.min(dim=1)
    minL[-1] = 0.0
    Cp = C-(minL.view(n, 1)@ones_m)
    T = min(10, 7.0*torch.log(torch.tensor(10.0))/torch.max(Cp))
    minC, _ = Cp.min(dim=0)
    minC[-1] = 0.0
    Cp = Cp-ones_n@minC.view(1, m)
    Cp = Cp/torch.max(Cp)
    return torch.exp(-T*Cp)


def franck_wolfe(x0, D, c, offset, kmax, n, m):
    k = 0
    converged = False
    x = x0
    T = 1.0
    dT = .5
    nb_iter = 40
    while (not converged) and (k <= kmax):
        Cp = (x.T@D+c).view(n+1, m+1)
        b = sinkhorn_diff(from_cost_to_similarity_exp(Cp),
                          nb_iter).view((n+1)*(m+1), 1)
        alpha = x.T@D@(b-x)+c.T@(b-x)
        # security check if b is not a local minima (does not occur with real hungarian)
        if alpha > 0:
            if k <= 3 and alpha >= 1e-3:
                sys.stderr.write(f"alpha positif('{k}') : {alpha.item()}")
            """
            if k==0:
                print('T=',T)
                print('Sim=',torch.exp(-T*Cp))
                print('Cp=',Cp)
                print('Cp original= ',(x.T@D+c).view(n+1,m+1))
            """
            return x

        beta = .5*(b-x).T@D@(b-x)
        if beta <= 0:
            t = 1.0
        else:
            t = min(-alpha/(2*beta), 1.0)
        x = x+t*(b-x)
        k = k+1
        converged = (-alpha < 1e-5)
        T = T+dT

    return x
