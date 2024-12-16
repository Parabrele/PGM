import numpy as np
import torch as th
from tqdm.auto import trange
device = "cuda" if th.cuda.is_available() else "cpu"


def ppca_closed_form(X, q, R=None):
    """
    Perform Probabilistic Principal Component Analysis on the dataset X.
    """
    X = X.to(device)
    if X.isnan().any():
        raise ValueError("X contains NaN values, use mppca")
    n, d = X.shape
    mu = X.mean(dim=0)
    X_centered = X - mu
    sigma = X_centered.T.cov()
    eigenvectors, eigenvalues, _ = th.svd(sigma)
    if q != d:
        sigma_ml = eigenvalues[q + 1 :].sum() / (d - q)
    else:
        sigma_ml = 0
    Uq = eigenvectors[:, :q]
    lambda_q = th.diag(eigenvalues[:q])
    id_q = th.eye(q).to(device)
    W_ML = Uq @ th.sqrt(lambda_q - sigma_ml * id_q)
    if R is not None:
        W_ML = W_ML @ R
    return W_ML


def mppca(X, q, k, epsilon=1e-4, max_iter=100):
    """
    Perform Mixtures of Probabilistic Principal Component Analyzers on the dataset X.

    Parameters:
    - X: Input data of shape (n, d)
    - q: Number of latent dimensions
    - k: Number of mixture components
    - epsilon: Convergence threshold for the EM algorithm

    Returns:
    - W: Weight matrices for each component
    - mu: Mean vectors for each component
    - sigma: Noise variance for each component
    """
    X = X.to(device)
    n, d = X.shape
    if q == d:
        raise ValueError("q must be less than d")
    W = th.randn(k, d, q).to(device)
    mu = th.randn(k, d).to(device)
    sigma = th.ones(k).to(device) * 0.1
    pi = th.ones(k).to(device) / k
    responsibilities = th.zeros(n, k).to(device)
    for i in trange(max_iter):
        M = W.transpose(1, 2) @ W + sigma.unsqueeze(-1).unsqueeze(-1) * th.eye(q).to(
            device
        ).unsqueeze(0)
        assert M.shape == (k, q, q), f"M shape: {M.shape}"
        M_inv = M.inverse()
        Z = th.einsum(
            "kqd, knd -> knq",
            M_inv @ W.transpose(1, 2),
            (X.unsqueeze(0) - mu.unsqueeze(1)),
        )
        assert Z.shape == (k, n, q), f"Z shape: {Z.shape}"
        # update responsibilities
        mu_normals = th.einsum("kdq, knq -> knd", W, Z) + mu.unsqueeze(
            1
        )  # todo gregoire??
        sigma_normals = sigma.unsqueeze(-1).unsqueeze(-1) * th.eye(d).to(device)
        assert sigma_normals.shape == (
            k,
            d,
            d,
        ), f"sigma_normals shape: {sigma_normals.shape}"
        prob = (
            -0.5
            * th.einsum(
                "knD, knD -> kn",
                th.einsum(
                    "knd, kdD -> knD",
                    (X.unsqueeze(0) - mu_normals),
                    sigma_normals.inverse(),
                ),
                (X.unsqueeze(0) - mu_normals),
            )
        ).exp() / ((2 * np.pi * sigma) ** (d / 2)).unsqueeze(1)
        assert prob.shape == (k, n), f"prob shape: {prob.shape}"
        responsibilities = prob * pi.unsqueeze(1)
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(dim=1, keepdim=True)
        # update parameters
        #  $\mathbf{M}_j = \mathbf{W}_j^{T}\mathbf{W}_j + \sigma_j^2\mathbf{I}$

        # todo: implement missing data?
        new_pi = responsibilities.mean(dim=1)
        new_mu = (responsibilities.unsqueeze(-1) * X.unsqueeze(0)).sum(
            dim=1
        ) / responsibilities.sum(dim=1, keepdim=True)
        new_sigmas = (
            th.einsum(
                "kNd, kND -> kdD",
                responsibilities.unsqueeze(-1) * (X - new_mu.unsqueeze(1)),
                (X - new_mu.unsqueeze(1)),
            )
        ) / (new_pi.unsqueeze(-1).unsqueeze(-1) * n)
        assert new_sigmas.shape == (k, d, d), f"new_sigmas shape: {new_sigmas.shape}"
        eigenvectors, eigenvalues, _ = th.svd(new_sigmas)
        sigma_ml = eigenvalues[:, q:].sum(dim=1) / (d - q)
        print(f"sigma_ml: {sigma_ml}")
        Uq = eigenvectors[:, :, :q]
        lambda_q = th.diag_embed(eigenvalues[:, :q])
        id_q = th.eye(q).unsqueeze(0).to(device)
        new_W = Uq @ th.sqrt(lambda_q - sigma_ml.unsqueeze(-1).unsqueeze(-1) * id_q)
        assert new_W.shape == (k, d, q), f"new_W shape: {new_W.shape}"
        update_size = ((new_W - W) ** 2).max()
        if update_size < epsilon:
            break
        W = new_W
        pi = new_pi
        mu = new_mu
        sigma = sigma_ml
    print(f"Finished in {i} iterations with update size {update_size}")
    return W, mu, sigma