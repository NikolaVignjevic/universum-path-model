# common.py
import numpy as np
import matplotlib.pyplot as plt

G_KPC_KM2_S2_MSUN = 4.30091e-6  # Gravitationskonstante in kpc*(km/s)^2 / Msun

# ------------------ Grid ------------------
def make_grid(nx, ny, dx_kpc):
    x = (np.arange(nx) - nx // 2) * dx_kpc
    y = (np.arange(ny) - ny // 2) * dx_kpc
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.hypot(X, Y)
    return X, Y, R

def radial_profile(Rmap, Fmap, nbins=80):
    rmax = Rmap.max()
    edges = np.linspace(0.0, rmax, nbins + 1)
    rc = 0.5 * (edges[1:] + edges[:-1])
    vals = np.zeros(nbins)
    idx = np.digitize(Rmap.ravel(), edges) - 1
    flat = Fmap.ravel()
    for i in range(nbins):
        m = idx == i
        if np.any(m):
            vals[i] = flat[m].mean()
    return rc, vals

# ------------------ Simple smoothers ------------------
def moving_average(y, w=3):
    if w <= 1:
        return y.copy()
    ypad = np.pad(y, (w//2, w//2), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode="same")[w//2: -w//2]

# ------------------ PDE Solvers (Gauss-Seidel) ------------------
def solve_stationary_deficit(nE, chi, Gamma, tau, D, dx, tol=1e-8, max_iter=6000):
    """
    (1/τ) δρ - D ∇² δρ = Γ (1-χ) nE  (stationär)
    Dirichlet-Rand: δρ = 0
    """
    invA = 1.0 / (1.0/tau + 4.0 * D / dx**2)
    delta = np.zeros_like(nE)
    for _ in range(max_iter):
        old = delta.copy()
        lap_nb = np.zeros_like(delta)
        lap_nb[1:-1, 1:-1] = (delta[2:, 1:-1] + delta[:-2, 1:-1] +
                              delta[1:-1, 2:] + delta[1:-1, :-2])
        rhs = Gamma * (1.0 - chi) * nE + (D / dx**2) * lap_nb
        delta[1:-1, 1:-1] = invA * rhs[1:-1, 1:-1]
        if np.max(np.abs(delta - old)) < tol * (np.max(np.abs(old)) + 1e-12):
            break
    return delta

def solve_screened_poisson(delta, ell_kpc, dx, tol=1e-8, max_iter=6000):
    """
    ρ_eff - ℓ² ∇² ρ_eff = δρ
    Dirichlet-Rand: ρ_eff = 0
    """
    ell2 = ell_kpc**2
    rho = np.zeros_like(delta)
    invA = 1.0 / (1.0 + 4.0 * ell2 / dx**2)
    for _ in range(max_iter):
        old = rho.copy()
        lap_nb = np.zeros_like(rho)
        lap_nb[1:-1, 1:-1] = (rho[2:, 1:-1] + rho[:-2, 1:-1] +
                              rho[1:-1, 2:] + rho[1:-1, :-2])
        rhs = delta + (ell2 / dx**2) * lap_nb
        rho[1:-1, 1:-1] = invA * rhs[1:-1, 1:-1]
        if np.max(np.abs(rho - old)) < tol * (np.max(np.abs(old)) + 1e-12):
            break
    return rho

def solve_poisson_2d_sigma(sigma_tot, dx, tol=1e-8, max_iter=12000):
    """
    ∇² Φ = 4π G Σ_tot  (dünne Scheiben-Näherung in 2D-Projektion)
    Dirichlet-Rand: Φ = 0
    """
    src = 4.0 * np.pi * G_KPC_KM2_S2_MSUN * sigma_tot
    phi = np.zeros_like(src)
    invA = 1.0 / (4.0 / dx**2)
    for _ in range(max_iter):
        old = phi.copy()
        lap_nb = np.zeros_like(phi)
        lap_nb[1:-1, 1:-1] = (phi[2:, 1:-1] + phi[:-2, 1:-1] +
                              phi[1:-1, 2:] + phi[1:-1, :-2])
        phi[1:-1, 1:-1] = invA * (lap_nb[1:-1, 1:-1] - dx**2 * src[1:-1, 1:-1])
        if np.max(np.abs(phi - old)) < tol * (np.max(np.abs(old)) + 1e-12):
            break
    return phi

# ------------------ Mapping Σ(R) → Σ(x,y) ------------------
def radial_to_map(Rmap, Rbins, SigmaR):
    """
    Robuste Abbildung eines radialen Profils SigmaR(Rbins) auf eine 2D-Karte:
    - entfernt NaNs/Inf
    - sortiert nach Radius
    - mittelt Duplikate gleicher Radien
    - stellt sicher: len(Rbins)==len(SigmaR), strikt aufsteigend
    """
    import numpy as np

    Rb = np.asarray(Rbins, dtype=float)
    Sr = np.asarray(SigmaR, dtype=float)

    # 1) Gültige Werte behalten
    m = np.isfinite(Rb) & np.isfinite(Sr)
    Rb = Rb[m]; Sr = Sr[m]

    # 2) Nur nicht-negative Radien
    mpos = Rb >= 0
    Rb = Rb[mpos]; Sr = Sr[mpos]

    # 3) Sortieren
    if Rb.size == 0:
        return np.zeros_like(Rmap)
    idx = np.argsort(Rb)
    Rb = Rb[idx]; Sr = Sr[idx]

    # 4) Duplikate mitteln
    Runiq, inv = np.unique(Rb, return_inverse=True)
    Sr_sum = np.zeros_like(Runiq, dtype=float)
    Sr_cnt = np.zeros_like(Runiq, dtype=float)
    for i, j in enumerate(inv):
        Sr_sum[j] += Sr[i]
        Sr_cnt[j] += 1.0
    Sr_avg = Sr_sum / np.maximum(Sr_cnt, 1.0)

    # 5) Edge cases
    if Runiq.size < 2:
        # zu wenig Stützstellen -> konstante Karte
        val = Sr_avg[0] if Sr_avg.size else 0.0
        return np.full_like(Rmap, val)

    # 6) Interpolation auf 2D-Karte
    Sigma_map = np.interp(Rmap.ravel(), Runiq, Sr_avg, left=Sr_avg[0], right=0.0)
    return Sigma_map.reshape(Rmap.shape)

# ------------------ Rotation curve from Φ ------------------
def rotation_curve_from_phi(Rmap, Phi, nbins=80):
    rc, phi_r = radial_profile(Rmap, Phi, nbins=nbins)
    # vorsichtiges Ableiten
    dphi = np.gradient(phi_r, rc, edge_order=2)
    vc2 = np.clip(rc * dphi, a_min=0.0, a_max=None)
    vc = np.sqrt(vc2)
    return rc, vc

# ------------------ Quick plotting ------------------
def quick_plot_maps(X, Y, rho_eff, out_png):
    plt.figure(figsize=(6,5))
    plt.imshow(rho_eff.T, origin="lower",
               extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.colorbar(label=r"$\rho_{\rm eff}$ (arb.)")
    plt.xlabel("x [kpc]"); plt.ylabel("y [kpc]")
    plt.title("Effektive Dichte $\\rho_{\\rm eff}$")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def quick_plot_rotation(rc, vc_model, rc_obs=None, vobs=None, out_png="rotation.png"):
    plt.figure(figsize=(6,4))
    plt.plot(rc, vc_model, lw=2, label="Model")
    if rc_obs is not None and vobs is not None:
        plt.scatter(rc_obs, vobs, s=12, alpha=0.7, label="Observed (SPARC)")
    plt.xlabel("r [kpc]"); plt.ylabel("v_c [km/s] (rel./km/s)")
    plt.title("Rotationskurve")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
