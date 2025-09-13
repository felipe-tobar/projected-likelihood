import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sphere_with_repulsion(d, M, n_iter=100, lr=0.05, seed=None):
    """
    Sample M unit vectors in R^d roughly uniformly on the sphere,
    using random init + repulsion dynamics.

    Args:
        d (int): dimension of ambient space
        M (int): number of vectors
        n_iter (int): number of repulsion steps
        lr (float): learning rate / step size for repulsion
        seed (int): random seed (optional)

    Returns:
        np.ndarray: shape (d, M), columns are unit vectors
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d, M))
    W /= np.linalg.norm(W, axis=0, keepdims=True)  # project to unit sphere

    for _ in range(n_iter):
        # Pairwise dot products
        G = W.T @ W  # (M, M)
        np.fill_diagonal(G, 0.0)

        # Repulsion force = push away proportional to dot product
        F = W @ G  # shape (d, M)

        # Update and renormalize
        W = W + lr * F
        W /= np.linalg.norm(W, axis=0, keepdims=True)

    return W

def repulsive_loss(Z, scale=1.0, min_dist=2.0):
    # Z: torch tensor shape (M, D)
    # scale: strength of repulsion
    # min_dist: preferred minimum distance (same units as X)
    M = Z.shape[0]
    # pairwise squared distances (M,M)
    diff = Z.unsqueeze(1) - Z.unsqueeze(0)        # (M,M,D)
    sqdist = (diff * diff).sum(dim=-1)            # (M,M)
    # ignore diagonal
    mask = ~torch.eye(M, dtype=torch.bool, device=Z.device)
    # penalize distances below min_dist using exp or inverse-power
    sigma2 = (min_dist/2.0)**2
    penalties = torch.exp(-sqdist / (2*sigma2)) * mask
    # sum over unique pairs and normalize
    loss = scale * penalties.sum() / (M * (M-1) / 2.0)
    return loss


def init_Z_kmeanspp(X_np, M, random_state=0):
    """
    Initialise inducing points with k-means++ cluster centers.

    Args:
        X_np : ndarray, shape (N, D)
            Training data (numpy array).
        M : int
            Number of inducing points.
        random_state : int
            Random seed for reproducibility.

    Returns:
        Z_init : ndarray, shape (M, D)
            Initial inducing points.
    """
    start = time.perf_counter()

    kmeans = KMeans(
        n_clusters=M,
        init="k-means++",
        n_init=10,
        random_state=random_state,
    )
    kmeans.fit(X_np)

    end = time.perf_counter()
    elapsed = end - start
    print(f"[init_Z_kmeanspp] KMeans finished in {elapsed:.3f} seconds")

    return kmeans.cluster_centers_



def plot_eigenvalues(K, ax=None, title="Eigenvalues of K"):
    """
    Plot eigenvalues of a symmetric positive semi-definite matrix K.

    Args:
        K : torch.Tensor, shape (N, N)
        ax : matplotlib axis (optional)
        title : str
    """
    # Ensure symmetric if numerical noise present
    K_sym = 0.5 * (K + K.T)

    # Compute eigenvalues
    eigvals = torch.linalg.eigvalsh(K_sym).detach().cpu().numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(eigvals, "o-", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    ax.grid(True)

    return ax, eigvals


def gaussian_window_weights(N, M, overlap_factor=2.5):
    # N: dimension of weight
    # M: number of weight vectors (Gaussian bumps)
    
    # positions in [0,1] for each column
    x = np.linspace(0, 1, N)
    
    # standard deviation so bumps don't overlap much
    sigma = 1 / (overlap_factor * M)
    
    W = np.zeros((N, M))
    for m in range(M):
        mu = m / M  # centre of m-th Gaussian
        W[:, m] = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        W[:, m] /= W[:, m].sum()   # normalize column (optional)
    
    return W

def random_one_hot_matrix(N, M, seed=None):
    if M > N:
        raise ValueError("M cannot be greater than N for unique one-hot positions.")
    
    rng = np.random.default_rng(seed)
    
    # randomly pick M unique row indices for the ones
    row_indices = rng.choice(N, size=M, replace=False)
    
    # create zero matrix
    W = np.zeros((N, M), dtype=np.float32)
    
    # assign ones
    W[row_indices, np.arange(M)] = 1.0
    
    return W

def split_into_three(M):
    base = M // 3
    remainder = M % 3

    parts = [base] * 3
    for i in range(remainder):
        parts[i] += 1

    return parts

def safe_cholesky(K, max_tries=6, initial_jitter=1e-6):
    """Attempt Cholesky decomposition with adaptive jitter.
    If all attempts fail, add a diagonal shift based on the most negative eigenvalue.
    """
    jitter = initial_jitter
    I = torch.eye(K.size(0), device=K.device, dtype=K.dtype)

    # Symmetrize to avoid numerical asymmetry
    K = 0.5 * (K + K.T)

    for i in range(max_tries):
        try:
            L = torch.linalg.cholesky(K + jitter * I)
            return L
        except RuntimeError as e:
            if "cholesky" in str(e).lower() or "not positive-definite" in str(e).lower():
                jitter *= 10
                print(f"[safe_cholesky] Failed, retrying with jitter={jitter:.1e}")
            else:
                raise

    # Eigenvalue-based fallback
    print("[safe_cholesky] Falling back to negative-eigenvalue shift")
    eigvals = torch.linalg.eigvalsh(K)
    
    min_eig = torch.min(eigvals)
    if min_eig < 0:
        shift = (-min_eig + 1e-8)  # small epsilon to make PD
        K += shift * I
        print(f"[safe_cholesky] Added diagonal shift {shift:.2e} based on min eigenvalue")
    L = torch.linalg.cholesky(K)

    return L

def safe_cholesky_experimental(K, max_tries=6, initial_jitter=1e-6):
    """Attempt Cholesky decomposition with adaptive jitter.
    - Runs on GPU if available
    - Falls back to CPU if running on MPS (since Cholesky isn't supported there)
    """
    # Detect if we're on MPS
    use_cpu = (K.device.type == "mps")

    if use_cpu:
        K = K.cpu()
    
    jitter = initial_jitter
    I = torch.eye(K.size(0), device=K.device, dtype=K.dtype)

    # Symmetrize to avoid numerical asymmetry
    K = 0.5 * (K + K.T)

    for i in range(max_tries):
        try:
            L = torch.linalg.cholesky(K + jitter * I)
            return L.to(K.device) if use_cpu else L
        except RuntimeError as e:
            if "cholesky" in str(e).lower() or "not positive-definite" in str(e).lower():
                jitter *= 10
                print(f"[safe_cholesky] Failed, retrying with jitter={jitter:.1e}")
            else:
                raise

    # Eigenvalue-based fallback
    print("[safe_cholesky] Falling back to negative-eigenvalue shift")
    eigvals = torch.linalg.eigvalsh(K)
    
    min_eig = torch.min(eigvals)
    if min_eig < 0:
        shift = (-min_eig + 1e-8)  # small epsilon to make PD
        K += shift * I
        print(f"[safe_cholesky] Added diagonal shift {shift:.2e} based on min eigenvalue")
    L = torch.linalg.cholesky(K)

    return L.to(torch.float32).to(K.device) if use_cpu else L




def plot_gp(
    x, mu, var, ax=None, 
    color_mean="b", color_fill="lightblue", label="GP mean", title = "GP posterior",
    X_obs=None, y_obs=None, color_obs="k",
    X_latent=None, y_latent=None, color_latent="r", marker_latent="."
):
    """
    Plot GP mean with ±2 stddev confidence interval, optionally including
    observed and latent/unobserved points.

    Args:
        x : 1D torch.Tensor, test inputs (N,)
        mu : 1D torch.Tensor, predictive mean (N,)
        var : 1D torch.Tensor, predictive variance (N,)
        ax : matplotlib axis (optional)
        color_mean : str, line color for mean
        color_fill : str, fill color for confidence interval
        label : str, label for the mean curve
        X_obs, y_obs : torch.Tensor, observed inputs and outputs (optional)
        color_obs : str, color for observed points
        X_latent, y_latent : torch.Tensor, latent/unobserved inputs and outputs (optional)
        color_latent : str, color for latent points
        marker_latent : str, marker style for latent points
    """
    # Sort test inputs
    sort_idx = torch.argsort(x.flatten())
    x_sorted = x[sort_idx]
    mu_sorted = mu[sort_idx]
    std_sorted = torch.sqrt(var[sort_idx])

    # Convert to numpy
    x_np = x_sorted.detach().cpu().numpy().ravel()
    mu_np = mu_sorted.detach().cpu().numpy().ravel()
    std_np = std_sorted.detach().cpu().numpy().ravel()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    # GP mean and uncertainty
    ax.plot(x_np, mu_np, color=color_mean, lw=2, label=label)
    ax.fill_between(x_np, mu_np - 2*std_np, mu_np + 2*std_np, 
                    color=color_fill, alpha=0.5, label="±2 std")
    ax.set_title(title)


    # Latent/unobserved points (if provided)
    if X_latent is not None and y_latent is not None:
        X_latent_np = X_latent.detach().cpu().numpy().ravel()
        y_latent_np = y_latent.detach().cpu().numpy().ravel()
        ax.plot(X_latent_np, y_latent_np, color_latent + marker_latent, 
                label="Latent points", markersize=5)

    # Observations (if provided)
    if X_obs is not None and y_obs is not None:
        X_obs_np = X_obs.detach().cpu().numpy().ravel()
        y_obs_np = y_obs.detach().cpu().numpy().ravel()
        ax.plot(X_obs_np, y_obs_np, color_obs + "x", label="Observations", markersize=4)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True)

    return ax




class GP(nn.Module):
    def __init__(self, X = None, y = None, kernel = 'SE', hypers = None, M=50, method = 'NLL'):
        super().__init__()
        #torch.manual_seed(12)
        #np.random.seed(12)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    device = torch.device("mps")  # Apple Silicon
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        self.kernel_ = kernel
        self.method = method
        self.M = M

        if X is not None and y is not None:
            # If 1D input (shape N x 1)
            if X.ndim == 2 and X.shape[1] == 1:
                sort_idx = torch.argsort(X[:, 0])
                sort_idx = sort_idx.to("cpu")   # make sure indices live on the same device as x

                #print(sort_idx.device)
                self.X = X[sort_idx]
                self.y = y[sort_idx]
            else:
                self.X = X
                self.y = y

            if method == 'NLL':
                1
            elif method == 'VFE':
                #perm = torch.randperm(X.shape[0])[:M]       # random indices
                #self.Z_ = nn.Parameter(X[perm].clone())      # choose those rows
                Z_np = init_Z_kmeanspp(X.cpu().numpy(), M)
                Z_t = torch.from_numpy(Z_np).to(self.X.device).to(self.X.dtype)
                self.Z_ = nn.Parameter(Z_t.clone())
                #self.Z_.requires_grad = False  # Freeze Z

            elif method == "proj-sphere":
                self.w = np.random.randn(y.shape[0], M)
                self.w /= np.linalg.norm(self.w, axis=0, keepdims=True)
            elif method == "proj-repulsive":
                w_0 = sphere_with_repulsion(y.shape[0], 400, n_iter=200, lr=0.1, seed=1)
                self.w = w_0[:,:M]
            elif method == "proj-orthogonal":
                self.w = np.random.randn(y.shape[0], M)
                # Orthogonalise via QR decomposition
                self.w, _ = np.linalg.qr(self.w)   # Q has orthonormal columns
            elif method == "proj-localised":
                self.w = gaussian_window_weights(y.shape[0], M, overlap_factor=3)
                self.w /= np.linalg.norm(self.w, axis=0, keepdims=True)
            elif method == "proj-onehot":
                self.w = random_one_hot_matrix(y.shape[0],M)
                self.w /= np.linalg.norm(self.w, axis=0, keepdims=True)
            elif method == "proj-mix":
                M1, M2, M3 = split_into_three(M)
                W1 = np.random.randn(y.shape[0], M1)
                W2 = gaussian_window_weights(y.shape[0], M2, overlap_factor=3)
                W3 = random_one_hot_matrix(y.shape[0],M3)
                self.w = np.concatenate([W1, W2, W3], axis=1)
                self.w /= np.linalg.norm(self.w, axis=0, keepdims=True)

            if method.startswith("proj"):
                self.w, _ = np.linalg.qr(self.w)  # orthogonalize columns
        
        #if hypers is None:
        #    if self.kernel_ == 'SE' or self.kernel_ == 'Laplace':
        #        self.hypers_ = nn.Parameter(torch.tensor(np.log([1.0, 1.0, 1.0])))
        #    elif self.kernel_ == 'RQ' or self.kernel_ == 'Per':
        #        self.hypers_ = nn.Parameter(torch.tensor(np.log([1.0, 1.0, 1.0, 1.0])))
        #    elif self.kernel_ == 'LocPer':
        #        self.hypers_ = nn.Parameter(torch.tensor(np.log([1.0, 1.0, 1.0, 1.0, 1.0])))
        #else:
        #    self.hypers_ = nn.Parameter(torch.tensor(np.log(hypers)))

        if hypers is None:
            if self.kernel_ == 'SE' or self.kernel_ == 'Laplace':
                h_np = np.log([1.0, 1.0, 1.0])
            elif self.kernel_ == 'RQ' or self.kernel_ == 'Per':
                h_np = np.log([1.0, 1.0, 1.0, 1.0])
            elif self.kernel_ == 'LocPer':
                h_np = np.log([1.0, 1.0, 1.0, 1.0, 1.0])
        else:
            h_np = np.log(hypers)


        h_t  = torch.from_numpy(h_np).to(device)
        self.hypers_ = nn.Parameter(h_t.clone())
    
    @property
    def hypers(self):
        raw = self.hypers_
        values = torch.exp(raw[:-1])                  # other hypers
        noise = torch.nn.functional.softplus(raw[-1]) + 1e-6  # always > 1e-6
        return torch.cat([values, noise.unsqueeze(0)])

    @property
    def Z(self):
        return self.Z_


    def compute_moments(self):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor.
        kernel = kernel"""
        X = self.X
        y = self.y

        # Kernel matrix (add jitter inside kernel_mat_self if needed)
        K = self.kernel_mat_self(X)

        # Cholesky factorisation (lower-triangular)
        L = safe_cholesky(K)

        # Solve using Cholesky 
        alpha = torch.cholesky_solve(y, L)   # (N, 1)
        self.L = L
        self.alpha = alpha
        self.K = K


    def forward(self, x):
        """compute prediction. fit() must have been called.
        x: test input data point. N x D tensor for the data dimensionality D."""
        # Kernel cross-covariance between training and test


        k = self.kernel_mat(self.X, x)    # (N, n*)

        # Solve triangular system 
        v = torch.cholesky_solve(k, self.L)   # equivalent to L⁻¹ k with reuse of Cholesky

        # Predictive mean
        mu = k.T @ self.alpha                 # (n*, 1)

        # Hyperparameters
        amplitude, *_, noise = self.hypers

        # Predictive variance
        var = amplitude + noise - (k * v).sum(dim=0)
        return mu, var

    def forward_VFE(self, x):
        """Posterior predictive mean and variance under Titsias' SGPR (VFE)."""

        # Inducing points
        Z = self.Z_

        # Covariance matrices
        Kuu = self.kernel_mat_self(Z)                          # (M, M)
        Kuf = self.kernel_mat(Z, self.X)                       # (M, N)
        Kfu = Kuf.T                                            # (N, M)
        Kuu_chol =safe_cholesky(Kuu)

        # Noise
        noise = self.hypers[-1]

        # A = Kuu + (1/sigma^2) Kuf Kfu
        A = Kuu + (1.0/noise) * (Kuf @ Kfu)
        A_chol = safe_cholesky(A)

        # alpha term = (1/sigma^2) Kfu y
        alpha = (1.0/noise) * Kuf @ self.y

        # Solve for c = A^{-1} alpha
        c = torch.cholesky_solve(alpha, A_chol)

        # Cross-covariance test-inducing
        Ksu = self.kernel_mat(x, Z)                            # (n*, M)

        # Predictive mean
        mu = Ksu @ c          # (n*, 1)

        # Predictive variance
        Kss = self.kernel_mat_self(x)                          # (n*, n*)
        v = torch.cholesky_solve(Ksu.T, Kuu_chol)  # (M, n*)
        w = torch.cholesky_solve(v, A_chol)        # (M, n*)

        Kss_diag = torch.diag(self.kernel_mat_self(x))  # shape (n*,)
        var = Kss_diag - torch.sum(Ksu * (v - w).T, dim=1)


        return mu, var




    def sample_from_prior(self, x, n_samples = 1, jitter=1e-4):
        """compute prediction. 
        x: test input data point. 
        N x D tensor for the data dimensionality D."""
        k = self.kernel_mat_self(x) + jitter*torch.eye(x.shape[0], device=x.device)  
        L = safe_cholesky(k)         # (D, D)
        z = torch.randn(k.size(0), n_samples, device=x.device, dtype=x.dtype)  # (D, n_samples), N(0, I)
        return L @ z 


    def nll(self, jitter=1e-3):
        """should be called before forward() call.
        X: training input data point. N x D tensor for the data dimensionality D.
        y: training target data point. N x 1 tensor.
        kernel = kernel"""
        X, y = self.X, self.y
        N = X.shape[0]
        K = self.kernel_mat_self(X) + 0*jitter * torch.eye(N, device=X.device)
        L = safe_cholesky(K)
        alpha = torch.cholesky_solve(y, L)  # cleaner than nested solve
            # log marginal likelihood
        log_marg_lik = (
            -0.5 * y.T @ alpha
            - torch.sum(torch.log(torch.diag(L))) ##this can be accelerated by -L.diagonal().log().sum()
            - 0.5 * N * np.log(2 * np.pi)
        )
        self.X = X
        self.y = y
        self.L = L
        self.alpha = alpha
        self.K = K
        return -log_marg_lik

    def projected_nll(self, w=None, jitter=1e-4):
        y = self.y
        np.random.seed(55)               # fix seed
        if self.method == 'proj-resample':
            M = self.M
            W1 = np.random.randn(y.shape[0], M)
            W2 = gaussian_window_weights(y.shape[0], M, overlap_factor=2)
            W3 = random_one_hot_matrix(y.shape[0],M)
            self.w = np.concatenate([W1, W2, W3], axis=1)
            self.w /= np.linalg.norm(self.w, axis=0, keepdims=True)
        w = self.w
        X = self.X
        w_torch = torch.from_numpy(w).to(y)   # match dtype/device of y
        y = w_torch.T @ y                # matrix-vector multiply
        D = y.shape[0]
        K = self.kernel_mat_self(X)  + jitter*torch.eye(X.shape[0], device=X.device)
        K = w_torch.T @ K @ w_torch
        L = safe_cholesky(K)
        alpha = torch.linalg.solve(L.T, torch.linalg.solve(L, y))
        marginal_likelihood = (-0.5 * y.T.mm(alpha) - torch.log(torch.diag(L)).sum() - D * 0.5 * np.log(2 * np.pi))
        return -marginal_likelihood



    def elbo(self,jitter=1e-3):
        X = self.X
        y = self.y
        Z = self.Z # define Z (inducing locs) somewhere
        #a, _ = torch.sort(Z.T)
        #a = torch.diff(a)
        #a = torch.min(a)
        #if a<0.5:
        #    print(f'Warning: inducing input are getting as close as {a.detach().numpy()}')
        N = X.shape[0]
        #log_ls, log_var, log_sig = self.log_lengthscale, self.log_variance, self.log_noise
        amplitude_scale = self.hypers[0]

        sigma2 = self.hypers[-1]

        # Kernels
        Kzz = self.kernel_mat_self(Z, add_noise = False)
        Kxz = self.kernel_mat(X,Z)
        diagKxx = amplitude_scale.expand(X.shape[0]) # [N]

        # Cholesky of Kzz
        Lz = safe_cholesky(Kzz)                  # Kzz = Lz Lz^T

        # Solve Kzz^{-1} Kzx via triangular solves
        # V = Kzz^{-1} Kzx = solve(Kzz, Kzx)
        Kzx = Kxz.T                                # [M,N]
        V = torch.cholesky_solve(Kzx, Lz)                # [M,N] ##CAMBIAR AQUI

        #TERM2
        # A = Kzz + (1/σ^2) Kzx Kxz = Kzz + (1/σ^2) Kzx @ Kxz
        A = Kzz + (Kzx @ Kxz) / sigma2
        #La = torch.linalg.cholesky(A)
        La = safe_cholesky(A)


        # log|B| = N log σ^2 - log|Kzz| + log|A|
        logdetKzz = 2.0 * torch.log(torch.diag(Lz)).sum()
        logdetA   = 2.0 * torch.log(torch.diag(La)).sum()
        logdetB   = N * torch.log(sigma2) - logdetKzz + logdetA

        #TERM3
        # B^{-1} y = (1/σ^2) [ y - (1/σ^2) Kxz A^{-1} Kzx y ]
        # First compute w = solve(A, Kzx y)
        Ky = Kzx @ y.squeeze(-1)                                   # [M]
        w  = torch.cholesky_solve(Ky.unsqueeze(-1), La).squeeze(-1)   # [M]
        Binv_y = (y.squeeze(-1) - (Kxz @ w) / sigma2) / sigma2

        quad = (y.squeeze(-1) * Binv_y).sum()


        #TERM4
        # trace term: tr(Kxx - Kxz Kzz^{-1} Kzx) / σ^2
        # tr(Kxz Kzz^{-1} Kzx) = tr(V Kxz) since V = Kzz^{-1} Kzx
        trace_Qff = (V * Kzx).sum()
        trace_term = (diagKxx.sum() - trace_Qff) / sigma2

        elbo = -0.5 * (N * torch.log(torch.tensor(2.0*np.pi, device=X.device)) + logdetB + quad + trace_term)
        lambda_repulse = 1000
        return elbo - lambda_repulse * repulsive_loss(self.Z_, scale=1.0, min_dist=1.0)
        #return elbo

    def kernel_mat_self(self, X, add_noise = True):
        if self.kernel_ == 'SE':
            amplitude_scale, length_scale, noise_scale = self.hypers
    
            sq = (X**2).sum(dim=1, keepdim=True)      # shape (N,1)
            sqdist = sq + sq.T - 2 * X @ X.T          # shape (N,N)
            
            K = amplitude_scale * torch.exp(-0.5 * sqdist / length_scale**2)
            
            # add noise on the diagonal
            if add_noise:
                K = K + noise_scale * torch.eye(len(X), device=X.device, dtype=X.dtype)

            return K

        elif self.kernel_ == 'Laplace':

            N = X.size(0)
            amplitude, length, noise = self.hypers

            # Compute pairwise Euclidean distances
            # torch.cdist returns L2 distances
            dist = torch.cdist(X, X, p=2)

            # Exponential kernel
            K = amplitude * torch.exp(- dist / length)

            # Add diagonal noise for numerical stability
            K = K + noise * torch.eye(N, device=X.device, dtype=X.dtype)

            return K

        elif self.kernel_ == 'RQ':
            N = X.size(0)
            amplitude, length, alpha, noise = self.hypers

            # Compute squared Euclidean distances 
            sq = (X**2).sum(dim=1, keepdim=True)      # shape (N,1)
            sqdist = sq + sq.T - 2 * X @ X.T          # shape (N,N)

            # Rational Quadratic kernel
            K = amplitude * (1 + sqdist / (2 * alpha * length**2)) ** (-alpha)

            # Add diagonal noise (match device and dtype)
            K = K + noise * torch.eye(N, device=X.device, dtype=X.dtype)

            return K
        elif self.kernel_ == 'Per':
            N = X.size(0)
            amplitude, length, period, noise = self.hypers

            # Compute pairwise Euclidean distances
            dist = torch.cdist(X, X, p=2)   # shape (N,N)

            # Periodic kernel
            arg = np.pi * dist / period
            K = amplitude * torch.exp(-2 * torch.sin(arg)**2 / length)

            # Add diagonal noise (match device and dtype)
            K = K + noise * torch.eye(N, device=X.device, dtype=X.dtype)

            return K

        elif self.kernel_ == 'LocPer':

            # Unpack hyperparameters
            amplitude, length_local, period, length_period, noise = self.hypers[:5]

            N = X.size(0)

            # Compute pairwise distances
            if X.size(1) == 1:
                # 1D case: faster than torch.cdist
                dist = torch.abs(X - X.T)        # shape (N, N)
            else:
                dist = torch.cdist(X, X, p=2)    # shape (N, N)

            # Locally periodic kernel = SE * Periodic
            se_part   = torch.exp(-0.5 * (dist**2) / (length_local**2))
            arg       = np.pi * dist / period
            per_part  = torch.exp(-2 * torch.sin(arg)**2 / (length_period**2))

            K = amplitude * se_part * per_part

            # Add diagonal noise for stability
            return  K + noise * torch.eye(N, device=X.device, dtype=X.dtype)



    def kernel_mat(self, X, Z):
        if self.kernel_ == 'SE':
            amplitude_scale, length_scale, _ = self.hypers
    
            # efficient pairwise squared distances
            # X: (N, D), Z: (M, D)
            X_norm = (X**2).sum(dim=1, keepdim=True)     # (N, 1)
            Z_norm = (Z**2).sum(dim=1, keepdim=True)     # (M, 1)

            # Compute pairwise squared distances efficiently
            sqdist = X_norm + Z_norm.T - 2 * X @ Z.T     # (N, M)
            
            return amplitude_scale * torch.exp(-0.5 * sqdist / length_scale**2)

        if self.kernel_ == 'RQ':
            # X: (N, D), Z: (M, D)
            amplitude, length, alpha, _ = self.hypers  # unpack hyperparameters

            # Compute squared Euclidean distances efficiently
            X_norm = (X**2).sum(dim=1, keepdim=True)   # (N,1)
            Z_norm = (Z**2).sum(dim=1, keepdim=True)   # (M,1)
            sqdist = X_norm + Z_norm.T - 2 * X @ Z.T   # (N,M)

            # Optional: clamp for numerical stability
            sqdist = torch.clamp(sqdist, min=0.0)

            # Rational Quadratic kernel
            return  amplitude * (1 + sqdist / (2 * alpha * length**2)) ** (-alpha)



        elif self.kernel_ == 'Laplace':
            amplitude, length, _ = self.hypers  # unpack hyperparameters

            # If X and Z are both 1D (common for time series), compute distances efficiently
            if X.size(1) == 1 and Z.size(1) == 1:
                dist = torch.abs(X - Z.T)        # shape (N, M)
            else:
                # General case: Euclidean distances
                dist = torch.cdist(X, Z, p=2)    # shape (N, M)

            # Laplace kernel
            return  amplitude * torch.exp(-dist / length)

        elif self.kernel_ == 'Per':
            # Unpack hyperparameters
            amplitude, length, period = self.hypers[:3]

            # Compute pairwise distances
            if X.size(1) == 1 and Z.size(1) == 1:
                # 1D case: faster than torch.cdist
                dist = torch.abs(X - Z.T)        # shape (N, M)
            else:
                dist = torch.cdist(X, Z, p=2)    # shape (N, M)

            # Periodic kernel
            arg = np.pi * dist / period
            return  amplitude * torch.exp(-2 * torch.sin(arg)**2 / length)

        elif self.kernel_ == 'LocPer':

            # Unpack hyperparameters
            amplitude, length_local, period, length_period = self.hypers[:4]

            # Compute pairwise distances
            if X.size(1) == 1 and Z.size(1) == 1:
                # 1D case: faster than torch.cdist
                dist = torch.abs(X - Z.T)        # shape (N, M)
            else:
                dist = torch.cdist(X, Z, p=2)    # shape (N, M)

            # Locally periodic kernel = SE * Periodic
            se_part   = torch.exp(-0.5 * (dist**2) / (length_local**2))
            arg       = np.pi * dist / period
            per_part  = torch.exp(-2 * torch.sin(arg)**2 / (length_period**2))

            return amplitude * se_part * per_part



    def train_step(self, obj = 'nll', opt_name = 'Adam', lr=0.01, n_steps=1, tol=1e-3, verbose =  False):
        """gradient-based optimization of hyperparameters
        opt: torch.optim.Optimizer object."""
        if verbose: 
            print(f'Optimising {obj} using {opt_name}')

        if opt_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        elif opt_name == "BFGS":
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr)

        # Training step
        prev_loss = np.inf
        no_improve_count = 0  # counter for consecutive non-improving steps
        max_no_improve = 5    # M consecutive steps to trigger early stopping (choose your M)

        print(f'norma: {self.hypers}')
            
        start = time.perf_counter()
        for epoch in range(n_steps):
            if opt_name == "BFGS":
                # LBFGS needs a closure
                def closure():
                    optimizer.zero_grad()
                    ## loss fn
                    if obj == 'nll':
                        loss = self.nll()
                    elif obj == 'elbo':
                        loss = -self.elbo()
                    elif obj == 'proj':
                        loss = self.projected_nll()
                    ##
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)

            else:
                if obj == 'nll':
                    loss = self.nll()
                elif obj == 'elbo':
                    loss = -self.elbo()
                elif obj == 'proj':
                    loss = self.projected_nll()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            
            if opt_name == "Adam" and epoch % 200 == 0 and verbose:
                print(f"Epoch {epoch}: loss = {loss.item():.4f}")
            if opt_name == "BFGS" and epoch % 10 == 0 and verbose:
                print(f"Epoch {epoch}: loss = {loss.item():.4f}")

            #######
            #if abs(prev_loss - loss.item()) < tol:
            #    if verbose: 
            #        print(f"Early stop at iteration {epoch}.")
            #    break
            #prev_loss = loss.item()
            ########

 

            # check improvement
            if abs(prev_loss - loss.item()) < tol:
                no_improve_count += 1
                if no_improve_count >= max_no_improve:
                    if verbose:
                        print(f"Early stop at iteration {epoch}: loss did not improve for {max_no_improve} steps.")
                    break
            else:
                no_improve_count = 0  # reset if improvement seen

            prev_loss = loss.item()


    
        end = time.perf_counter()     # end timer
        elapsed = float(end - start)
        achieved_nll = float(self.nll().detach().cpu().numpy().flatten())
        if verbose: 
            print(f"Elapsed time: {elapsed:.1f}[s] with NLL = {achieved_nll:.2f}")

        return {
            "loss": loss.item(),
            "time": elapsed,
            "nll" : achieved_nll
        }


        ## The following part is not working and not used, fix it if training trajectories needed
        if False:
            if self.kernel_ == 'SE' or self.kernel_ == 'Laplace':
                return {
                "loss": loss.item(),
                "length": self.hypers[1].detach().cpu(),
                "noise": self.hypers[2].detach().cpu(),
                "amplitude": self.hypers[0].detach().cpu(),
                #"Z": self.Z.detach().cpu(),
                }
            elif self.kernel_ == 'Per':
                return {
                "loss": loss.item(),
                "length": self.hypers[1].detach().cpu(),
                "noise": self.hypers[3].detach().cpu(),
                "period": self.hypers[2].detach().cpu(),
                "amplitude": self.hypers[0].detach().cpu(),
                }
            elif self.kernel_ == 'RQ':
                return {
                "loss": loss.item(),
                "length": self.hypers[1].detach().cpu(),
                "noise": self.hypers[3].detach().cpu(),
                "alpha": self.hypers[2].detach().cpu(),
                "amplitude": self.hypers[0].detach().cpu(),
                }
