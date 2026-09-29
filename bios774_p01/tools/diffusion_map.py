"""
Diffusion Map (Coifman & Lafon, 2006), following the construction in the
BIOS 774 lecture notes (L7-8, Section 9.1).

Steps, in the notes' notation:
  1. Gaussian affinities on a symmetrized k-NN graph
         W_ij = exp(-||x_i - x_j||^2 / eps)        (0 if not neighbors)
  2. Density normalization with alpha in [0, 1]
         W^(alpha) = D^-alpha  W  D^-alpha,        D_ii = sum_j W_ij
  3. Row-normalize into a Markov transition matrix
         P^(alpha) = (D^(alpha))^-1 W^(alpha)
  4. Eigendecompose P^(alpha): eigenvalues 1 = lam_0 >= lam_1 >= ...,
     drop the constant eigenvector psi_0, and embed with
         Z = [lam_1^t psi_1, ..., lam_d^t psi_d]
     where the psi's are normalized so sum_i psi(i)^2 pi_i = 1.
     With that normalization, Euclidean distance in Z approximates the
     diffusion distance D_t (Section 9.2).

alpha:  0   -> same operator as Laplacian Eigenmaps (density fully kept)
        0.5 -> Fokker-Planck / Langevin diffusion
        1   -> Laplace-Beltrami: intrinsic geometry, density removed (default)
t:      diffusion time; larger t emphasizes coarser, longer-range structure.

Practical note: the notes use a full Gaussian kernel over all pairs. Here the
kernel is truncated to a k-NN graph so memory stays O(n*k) instead of O(n^2),
which is required for PathMNIST-sized inputs. The Gaussian weights are
negligible beyond the nearest neighbors anyway when eps is set from the
k-NN distances.
"""
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


class DiffusionMap:
    def __init__(self, n_components=2, n_neighbors=15, alpha=1.0, t=1,
                 epsilon=None, random_state=0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if int(t) != t or t < 0:
            raise ValueError("t must be a non-negative integer")
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.t = int(t)
        self.epsilon = epsilon
        self.random_state = random_state

    def fit_transform(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        d = self.n_components

        # 1. k-NN graph with Gaussian weights
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        dist, idx = nn.kneighbors(X)
        dist, idx = dist[:, 1:], idx[:, 1:]          # drop self
        sq = dist ** 2
        # Bandwidth: median squared k-NN distance, unless given.
        eps = float(np.median(sq)) if self.epsilon is None else float(self.epsilon)
        if eps <= 0:
            raise ValueError("epsilon must be positive (duplicate points?)")
        rows = np.repeat(np.arange(n), k)
        W = sparse.csr_matrix((np.exp(-sq.ravel() / eps), (rows, idx.ravel())),
                              shape=(n, n))
        W = W.maximum(W.T)                            # symmetrize
        W.setdiag(1.0)                                # exp(0) self-affinity

        # 2. Density normalization  W^(alpha) = D^-a W D^-a
        deg = np.asarray(W.sum(axis=1)).ravel()
        Da = sparse.diags(deg ** -self.alpha)
        Wa = Da @ W @ Da

        # 3-4. P = (D^(a))^-1 W^(a). Solve via the symmetric matrix
        #      S = (D^(a))^-1/2 W^(a) (D^(a))^-1/2, which has the same
        #      eigenvalues; psi = (D^(a))^-1/2 phi.
        deg_a = np.asarray(Wa.sum(axis=1)).ravel()
        inv_sqrt = sparse.diags(1.0 / np.sqrt(deg_a))
        S = inv_sqrt @ Wa @ inv_sqrt
        S = (S + S.T) / 2                             # remove round-off asymmetry

        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(n)
        vals, vecs = eigsh(S, k=d + 1, which="LA", v0=v0)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # A disconnected k-NN graph gives several eigenvalues equal to 1;
        # the embedding then just indexes connected components.
        n_unit = int(np.sum(vals > 1 - 1e-8))
        if n_unit > 1:
            warnings.warn(
                f"k-NN graph appears disconnected ({n_unit} eigenvalues ~ 1). "
                "Leading coordinates will mostly separate graph components; "
                "consider increasing n_neighbors."
            )

        pi = deg_a / deg_a.sum()                      # stationary distribution
        psi = vecs / np.sqrt(pi)[:, None]             # sum_i psi^2 pi_i = 1
        # Fix sign for reproducibility: largest-|value| entry positive.
        signs = np.sign(psi[np.abs(psi).argmax(axis=0), np.arange(psi.shape[1])])
        psi = psi * signs

        self.eigenvalues_ = vals[1:d + 1]
        self.epsilon_ = eps
        self.embedding_ = psi[:, 1:d + 1] * (self.eigenvalues_ ** self.t)
        return self.embedding_
