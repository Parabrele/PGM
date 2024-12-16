import numpy as np
import torch as th
from tqdm.auto import trange
from sklearn.cluster import KMeans

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


def mppca_full(X, q, k, epsilon=1e-4, max_iter=100, use_kmeans_init=True):
    """
    Perform Mixtures of Probabilistic Principal Component Analyzers on the dataset X.

    Parameters:
    - X: Input data of shape (n, d)
    - q: Number of latent dimensions
    - k: Number of mixture components
    - epsilon: Convergence threshold for the EM algorithm
    - max_iter: Maximum number of iterations
    - use_kmeans_init: Whether to initialize using K-means (True) or random initialization (False)

    Returns:
    - W: Weight matrices for each component
    - mu: Mean vectors for each component
    - sigma: Noise variance for each component
    """
    X = X.to(device)
    n, d = X.shape
    if q == d:
        raise ValueError("q must be less than d")

    if use_kmeans_init:
        # Initialize using K-means
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X.cpu().numpy())

        # Initialize parameters based on k-means results
        mu = th.tensor(kmeans.cluster_centers_, device=device)
        pi = th.tensor([np.mean(clusters == i) for i in range(k)], device=device)

        # Initialize W using PCA on each cluster
        W = th.zeros(k, d, q, device=device)
        sigma_squared = th.zeros(k, device=device)

        for i in range(k):
            cluster_points = X[clusters == i]
            if len(cluster_points) > 0:
                # Compute cluster-specific PCA
                cluster_centered = cluster_points - cluster_points.mean(0)
                try:
                    U, S, _ = th.svd(cluster_centered.T @ cluster_centered)
                    W[i] = U[:, :q] * th.sqrt(S[:q] / len(cluster_points)).unsqueeze(0)
                    # Initialize sigma as the mean of unused eigenvalues
                    sigma_squared[i] = S[q:].mean() / (d - q) if q < d else 1.0
                except Exception as e:
                    print(f"Error in SVD: {e}")
                    # Fallback initialization if SVD fails
                    W[i] = th.randn(d, q, device=device) * 0.01
                    sigma_squared[i] = 1.0
            else:
                # Fallback for empty clusters
                W[i] = th.randn(d, q, device=device) * 0.01
                sigma_squared[i] = 1.0
    else:
        # Random initialization
        mu = X[th.randperm(n)[:k]]  # Random subset of data points as means
        pi = th.ones(k, device=device) / k  # Uniform mixture weights
        W = th.randn(k, d, q, device=device) * 0.01  # Random initialization of W
        sigma_squared = th.ones(k, device=device)  # Initialize all sigmas to 1

    # Main EM loop
    for i in trange(max_iter):
        M = W.transpose(1, 2) @ W + sigma_squared.unsqueeze(-1).unsqueeze(-1) * th.eye(
            q
        ).to(device).unsqueeze(0)
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
        sigma_normals = sigma_squared.unsqueeze(-1).unsqueeze(-1) * th.eye(d).to(device)
        assert sigma_normals.shape == (
            k,
            d,
            d,
        ), f"sigma_normals shape: {sigma_normals.shape}"
        # Calculate log responsibilities
        log_resp = (
            th.log(pi).unsqueeze(1)  # [k, 1]
            - 0.5 * d * th.log(2 * np.pi * sigma_squared).unsqueeze(1)  # [k, 1]
            - 0.5
            * th.einsum(
                "knD,knD->kn",
                th.einsum(
                    "knd,kdD->knD",
                    (X.unsqueeze(0) - mu_normals),
                    sigma_normals.inverse(),
                ),
                (X.unsqueeze(0) - mu_normals),
            )
        )

        # Numerical stability: subtract max and exp
        log_resp_max = log_resp.max(dim=0, keepdim=True)[0]
        responsibilities = (log_resp - log_resp_max).exp()
        responsibilities /= responsibilities.sum(dim=0, keepdim=True)
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
        update_size = max(
            ((new_W - W) ** 2).max(),
            ((new_pi - pi) ** 2).max(),
            ((new_mu - mu) ** 2).max(),
            ((sigma_ml - sigma_squared) ** 2).max(),
        )
        if update_size < epsilon:
            break
        W = new_W
        pi = new_pi
        mu = new_mu
        sigma_squared = sigma_ml
    print(f"Finished in {i} iterations with update size {update_size}")
    return W, mu, sigma_squared


def compute_Z(W, X, mu, sigma_squared, q, device=device):
    """Compute latent variables Z."""
    M = W.transpose(1, 2) @ W + sigma_squared.unsqueeze(-1).unsqueeze(-1) * th.eye(
        q
    ).to(device).unsqueeze(0)
    M_inv = M.inverse()
    return th.einsum(
        "kqd, knd -> knq", M_inv @ W.transpose(1, 2), (X.unsqueeze(0) - mu.unsqueeze(1))
    )


def compute_responsibilities(X, W, mu, sigma_squared, pi, d, device):
    """Compute responsibilities for each component."""
    # Compute normal distribution parameters
    Z = compute_Z(W, X, mu, sigma_squared, W.shape[-1], device)
    mu_normals = th.einsum("kdq, knq -> knd", W, Z) + mu.unsqueeze(1)
    sigma_normals = sigma_squared.unsqueeze(-1).unsqueeze(-1) * th.eye(d).to(device)

    # Calculate log responsibilities
    log_resp = (
        th.log(pi).unsqueeze(1)
        - 0.5 * d * th.log(2 * np.pi * sigma_squared).unsqueeze(1)
        - 0.5
        * th.einsum(
            "knD,knD->kn",
            th.einsum(
                "knd,kdD->knD",
                (X.unsqueeze(0) - mu_normals),
                sigma_normals.inverse(),
            ),
            (X.unsqueeze(0) - mu_normals),
        )
    )

    # Numerical stability: subtract max and exp
    log_resp_max = log_resp.max(dim=0, keepdim=True)[0]
    responsibilities = (log_resp - log_resp_max).exp()
    responsibilities /= responsibilities.sum(dim=0, keepdim=True)

    return responsibilities


def update_parameters(X, responsibilities, d, q, device):
    """Update model parameters based on responsibilities."""
    n = X.shape[0]
    new_pi = responsibilities.mean(dim=1)

    # Update means
    new_mu = (responsibilities.unsqueeze(-1) * X.unsqueeze(0)).sum(
        dim=1
    ) / responsibilities.sum(dim=1, keepdim=True)

    # Update sigmas
    new_sigmas = (
        th.einsum(
            "kNd, kND -> kdD",
            responsibilities.unsqueeze(-1) * (X - new_mu.unsqueeze(1)),
            (X - new_mu.unsqueeze(1)),
        )
    ) / (new_pi.unsqueeze(-1).unsqueeze(-1) * n)

    # Compute new W and sigma_ml
    eigenvectors, eigenvalues, _ = th.svd(new_sigmas)
    sigma_ml = eigenvalues[:, q:].sum(dim=1) / (d - q)
    Uq = eigenvectors[:, :, :q]
    lambda_q = th.diag_embed(eigenvalues[:, :q])
    id_q = th.eye(q).unsqueeze(0).to(device)
    new_W = Uq @ th.sqrt(lambda_q - sigma_ml.unsqueeze(-1).unsqueeze(-1) * id_q)

    return new_W, new_mu, new_pi, sigma_ml


def initialize_with_kmeans(X, k, q, d):
    """Initialize parameters using K-means clustering."""
    n = X.shape[0]
    # Initialize using K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X.cpu().numpy())

    # Initialize parameters based on k-means results
    mu = th.tensor(kmeans.cluster_centers_, device=device)
    pi = th.tensor([np.mean(clusters == i) for i in range(k)], device=device)

    # Initialize W using PCA on each cluster
    W = th.zeros(k, d, q, device=device)
    sigma_squared = th.zeros(k, device=device)

    for i in range(k):
        cluster_points = X[clusters == i]
        if len(cluster_points) > 0:
            # Compute cluster-specific PCA
            cluster_centered = cluster_points - cluster_points.mean(0)
            try:
                U, S, _ = th.svd(cluster_centered.T @ cluster_centered)
                W[i] = U[:, :q] * th.sqrt(S[:q] / len(cluster_points)).unsqueeze(0)
                # Initialize sigma as the mean of unused eigenvalues
                sigma_squared[i] = S[q:].mean() / (d - q) if q < d else 1.0
            except Exception as e:
                print(f"Error in SVD: {e}")
                # Fallback initialization if SVD fails
                W[i] = th.randn(d, q, device=device) * 0.01
                sigma_squared[i] = 1.0
        else:
            # Fallback for empty clusters
            W[i] = th.randn(d, q, device=device) * 0.01
            sigma_squared[i] = 1.0

    return W, mu, pi, sigma_squared


def initialize_randomly(X, k, q, d):
    """Initialize parameters randomly."""
    n = X.shape[0]
    # Random initialization
    mu = X[th.randperm(n)[:k]]  # Random subset of data points as means
    pi = th.ones(k, device=device) / k  # Uniform mixture weights
    W = th.randn(k, d, q, device=device) * 0.01  # Random initialization of W
    sigma_squared = th.ones(k, device=device)  # Initialize all sigmas to 1

    return W, mu, pi, sigma_squared


def mppca(X, q, k, epsilon=1e-4, max_iter=100, use_kmeans_init=True):
    """
    Perform Mixtures of Probabilistic Principal Component Analyzers on the dataset X.
    """
    X = X.to(device)
    n, d = X.shape
    if q == d:
        raise ValueError("q must be less than d")

    # Initialize parameters
    W, mu, pi, sigma_squared = (
        initialize_with_kmeans(X, k, q, d)
        if use_kmeans_init
        else initialize_randomly(X, k, q, d)
    )

    # Main EM loop
    for i in trange(max_iter):
        # E-step: compute responsibilities
        responsibilities = compute_responsibilities(
            X, W, mu, sigma_squared, pi, d, device
        )

        # M-step: update parameters
        new_W, new_mu, new_pi, sigma_ml = update_parameters(
            X, responsibilities, d, q, device
        )

        # Check convergence
        update_size = max(
            ((new_W - W) ** 2).max(),
            ((new_pi - pi) ** 2).max(),
            ((new_mu - mu) ** 2).max(),
            ((sigma_ml - sigma_squared) ** 2).max(),
        )

        if update_size < epsilon:
            
            break

        W, pi, mu, sigma_squared = new_W, new_pi, new_mu, sigma_ml

    print(f"Finished in {i} iterations with update size {update_size}")
    return W, mu, sigma_squared, pi
