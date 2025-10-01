# run_ngc3198.py
import os
import numpy as np
from common import (make_grid, radial_to_map,
                    solve_stationary_deficit, solve_screened_poisson, solve_poisson_2d_sigma,
                    rotation_curve_from_phi, quick_plot_maps, quick_plot_rotation)
from sparc_loader import load_sparc_csv

# ---------------- Parameters ----------------
gal_name = "NGC3198"
csv_path = os.path.join("data", "sparc_ngc3198.csv")  # <- deine SPARC-CSV

# Gitter
nx = ny = 512
dx_kpc = 0.5  # Zellgröße

# Spur-Parameter
ell_kpc = 5.0
tau_gyr = 5.0
D_kpc2_per_gyr = 0.3
Gamma_W = 1.0
chi_const = 0.3  # innen später ggf. höher setzen

# M/L-Skalierung der SPARC-stellaren Beiträge
a_disk = 0.5
a_bulge = 0.7

# ---------------- Load SPARC & build Σ_baryon ----------------
R_sparc, Vobs, Sigma_R = load_sparc_csv(csv_path, a_disk=a_disk, a_bulge=a_bulge, smooth_w=5)

# 2D-Karte aufspannen
X, Y, Rmap = make_grid(nx, ny, dx_kpc)
print("len(R_sparc)=", len(R_sparc), " len(Sigma_R)=", len(Sigma_R))
print("R_sparc sorted?", np.all(np.diff(np.sort(R_sparc)) >= 0))
Sigma_b = radial_to_map(Rmap, R_sparc, Sigma_R)  # Msun/kpc^2 (projiziert)

# Quellterm n_E & Kohärenz χ
nE = Sigma_b / (Sigma_b.max() + 1e-30)
chi = np.full_like(nE, chi_const, dtype=float)

# ---------------- PDE-Kette ----------------
delta = solve_stationary_deficit(nE, chi, Gamma_W, tau_gyr, D_kpc2_per_gyr, dx_kpc)
rho_eff = solve_screened_poisson(delta, ell_kpc, dx_kpc)
Sigma_tot = Sigma_b + rho_eff
Phi = solve_poisson_2d_sigma(Sigma_tot, dx_kpc)

# Rotationskurve
rc, vc_model = rotation_curve_from_phi(Rmap, Phi, nbins=80)

# ---------------- Plots ----------------
os.makedirs("figures", exist_ok=True)
quick_plot_maps(X, Y, rho_eff, out_png=os.path.join("figures", f"{gal_name}_rho_eff.png"))

# Optional: beobachtete Kurve anlegen, falls Vobs vorhanden
rc_obs = R_sparc if Vobs is not None else None
quick_plot_rotation(rc, vc_model, rc_obs=rc_obs, vobs=Vobs,
                    out_png=os.path.join("figures", f"{gal_name}_rotation.png"))

print(f"Done: figures/{gal_name}_rho_eff.png, figures/{gal_name}_rotation.png")
