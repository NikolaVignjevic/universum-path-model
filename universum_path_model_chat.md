# Universum Path Model — Chat-Transcript (Markdown)

> **Hinweis:** Dies ist eine saubere, repo-taugliche Markdown-Fassung unseres Gesprächs.
> Sie enthält alle wesentlichen Inhalte (Annahmen, Axiome, Formeln, Experimente, Operatoren,
> Methoden, Datenquellen, Code) in strukturierter Reihenfolge – nicht den wörtlichen 1:1-Chat,
> sondern eine vollständige inhaltliche Transkription.

---

## Inhalt
1. Ausgangsfrage & Rahmen
2. Entropie, Poincaré-Rekurrenz, Boltzmann-Fluktuationen (kurz)
3. Faden-/Stoßwellen-Modell — Axiome (Cheat Sheet)
4. Gedankenexperimente
   - CHSH / Bell-Korrelationen (Singulett, Tsirelson)
   - Doppelspalt (Superposition, Dekohärenz)
5. Operatoren-Toolbox für die Stoßwelle
6. „Known Weaknesses & Open Questions“
7. Gravitation & Dunkle Materie als Bulk–Stoßwellen-Effekt
8. EEP (Einstein’sches Äquivalenzprinzip) — Klarstellung
9. Minimalmodell (PDE-Kette) & Methods (formelarm)
10. Datenquellen & Machbarkeit (SPARC etc.)
11. Projekt-Setup (Repo, README, Lizenz)
12. Python-Dateien (finale Versionen)
13. Starterpaket (ZIP) – Verweis
14. To‑do / Roadmap

---

## 1) Ausgangsfrage & Rahmen
- Kann man „der absoluten Entropie entkommen“? (physikalisch)
- Diskussion: 2. HS der Thermodynamik ist **statistisch**; Rekurrenzzeiten vs. beobachtbare Zeitskalen; makro- vs. mikro-Entropie.
- Fokuswechsel: **Modell der Realität** als **superfluide Stoßwelle** in einem **Bulk-Universum**,
  die **Fäden** (1D-Objekte) und **Knoten** (Kreuzungen = Ereignisse/Teilchen) erzeugt.
- Ziel: Quantenphänomene, Gravitation und (effektiv) Dunkle Materie konsistent aus dieser Dynamik deuten.

---

## 2) Entropie (kurz)
- Entropie als **statistische** Größe; auf endlichen Phasenräumen existiert **Poincaré-Rekurrenz**, aber astronomisch große Zeiten.
- Für praktische Physik: Entropie steigt **typisch**, nicht absolut determiniert; lokale Abnahmen bei globaler Zunahme möglich.

---

## 3) Faden-/Stoßwellen-Modell — Axiome (Cheat Sheet)
**(F1) Fäden:** unendliche 1D-Objekte \( f\in\mathcal{F} \) mit transversaler Schwingung \( \Psi_f(u,t)\in\mathbb{C}^d \).  
**(F2) Knoten/Ereignisse:** beobachtete „Teilchen“ = Kreuzungen; lokaler 2D-Modenraum \(|0\rangle,|1\rangle\).  
**(F3) Verschränkung:** gemeinsame Fadenbindung \( f \) & konservierte Fadenladung \( Q_f \) → Singulett \( |\Psi^-\rangle \).  
**(F4) Stoßwelle \(W\):** unitäre Umlagerung der Kreuzungsmatrix \(C\): \( C(t+dt)=\mathcal{O}_W C(t) \).  
**(F5) Emergenz:** Raumzeitabstände sind Projektionen der Bulk-Verknüpfung.

---

## 4) Gedankenexperimente

### 4.1 CHSH / Bell
Messungen entlang Achsen \( \hat a,\hat b \):  
\(\displaystyle E(\hat a,\hat b)=-\,\hat a\cdot\hat b,\quad |S|=2\sqrt{2}\) (Tsirelson).  
Interpretation: Korrelationen entstehen geometrisch durch **gemeinsamen Fadenindex**.

### 4.2 Doppelspalt
Zustand: \( |\psi\rangle=\alpha e^{i\phi_1}|p_1\rangle+\beta e^{i\phi_2}|p_2\rangle \).  
Dekohärenz durch Umgebungsknoten \( |e_1\rangle,|e_2\rangle \): Sichtbarkeit \(V=|\langle e_1|e_2\rangle|\).  
Messung = **Knoten-Neukopplung** durch \( \mathcal{O}_W \).

---

## 5) Operatoren-Toolbox \( \mathcal{O}_W \)
- **\(\mathcal{C}^\dagger\)**: Kreuzung erzeugen (Mess-/Teilchenerzeugung).  
- **\(\mathcal{C}\)**: Kreuzung löschen.  
- **\(\mathcal{R}\)**: Umleiten/Reconection (Streuung, Dekohärenz).  
- **\(\mathcal{S}\)**: Superposition (z. B. Doppelspalt).  
- **\(\mathcal{E}\)**: Verschränkung (Bindung über \(Q_f\)).

---

## 6) Known Weaknesses & Open Questions (Auszug)
**Konzeptuell:** Natur der Fäden, Eigenschaften des Bulks, Zeitpfeil.  
**Mathematisch:** exakte Operator-Algebra, kontinuierliche Dynamik, natürliche Entropiemetrik.  
**Experimentell:** neue, messbare Abweichungen? CMB/ISW-Limits? Bullet-Cluster-Morphologie?

---

## 7) Gravitation & Dunkle Materie als Bulk–Stoßwellen-Effekt
- Vor der ersten Welle: Bulk = skalares, superfluides Medium.  
- Stoßwelle **entnimmt** lokal Feld („Material“) → **Defizite** \( \delta\rho_\phi \).  
- Defizite persistieren/relaxen/diffundieren → **Spur** \( \rho_{\rm eff} \) (DM-ähnlich).  
- Gravitation = Potential aus \( \rho_{\text{mass}}+\rho_{\rm eff} \).  
- Bullet-Cluster-kompatibel, wenn Spur an collisionless Bahnen haftet (Persistenz \(\tau\) groß).

---

## 8) EEP — Klarstellung
- **Einstein’sches Äquivalenzprinzip (EEP)**: WEP + LLI + LPI.  
- In diesem Modell: **Photonen, normale Materie, BEC** fallen im selben \( \Phi \).  
- Unterschied nur in der **Spurproduktion**: Kohärente Zustände (BEC) erzeugen **weniger** \( \delta\rho_\phi \) (Kohärenzfaktor \( \chi\to1 \)), gravitiert wird **voll** über \(\rho_{\text{mass}}\).

---

## 9) Minimalmodell (PDE-Kette) & Methods

**Stationär:**  
\(\displaystyle \frac{1}{\tau}\,\delta\rho_\phi - D\,\nabla^2\delta\rho_\phi = \Gamma_W (1-\chi)\,n_E\)  
\(\displaystyle \rho_{\rm eff} - \ell^2 \nabla^2 \rho_{\rm eff} = \delta\rho_\phi\)  
\(\displaystyle \nabla^2 \Phi = 4\pi G\,(\rho_{\text{mass}}+\rho_{\rm eff})\)

**Methoden (formelarm):** 2D-Gitter, Gauss–Seidel-Relaxation, robuste Interpolationen, Open-Boundary groß wählen, Rotationskurve \(v_c(r)=\sqrt{r\,\partial_r\Phi}\), Lensing via \(\Sigma,\kappa,\gamma_t\).

---

## 10) Datenquellen & Machbarkeit
- **SPARC**: Rotationskurven + 3.6 µm-Photometrie (Start mit NGC 3198, NGC 2403).  
- **SLACS/KiDS/DES**: Lensing (später).  
- Heimrechner ausreichend: 256–512² Gitter, Sekunden bis Minuten Laufzeit mit einfachen Solvern.

---

## 11) Projekt-Setup (Repo)
- Name: `universum-path-model`  
- Beschreibung (kurz): *Minimalmodell … Spur (δρφ→ρ_eff→Φ); Rotationskurven & Lensing; EEP-konform.*  
- Sichtbarkeit: Public; `.gitignore: Python`; Lizenz: Apache‑2.0.

---

## 12) Python-Dateien (finale Versionen)

### 12.1 `common.py`
```python
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

def moving_average(y, w=3):
    if w <= 1:
        return y.copy()
    ypad = np.pad(y, (w//2, w//2), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(ypad, kernel, mode="same")[w//2: -w//2]

def solve_stationary_deficit(nE, chi, Gamma, tau, D, dx, tol=1e-8, max_iter=6000):
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

def radial_to_map(Rmap, Rbins, SigmaR):
    import numpy as np
    Rb = np.asarray(Rbins, dtype=float)
    Sr = np.asarray(SigmaR, dtype=float)
    m = np.isfinite(Rb) & np.isfinite(Sr)
    Rb = Rb[m]; Sr = Sr[m]
    mpos = Rb >= 0
    Rb = Rb[mpos]; Sr = Sr[mpos]
    if Rb.size == 0:
        return np.zeros_like(Rmap)
    idx = np.argsort(Rb)
    Rb = Rb[idx]; Sr = Sr[idx]
    Runiq, inv = np.unique(Rb, return_inverse=True)
    Sr_sum = np.zeros_like(Runiq, dtype=float)
    Sr_cnt = np.zeros_like(Runiq, dtype=float)
    for i, j in enumerate(inv):
        Sr_sum[j] += Sr[i]; Sr_cnt[j] += 1.0
    Sr_avg = Sr_sum / np.maximum(Sr_cnt, 1.0)
    if Runiq.size < 2:
        val = Sr_avg[0] if Sr_avg.size else 0.0
        return np.full_like(Rmap, val)
    Sigma_map = np.interp(Rmap.ravel(), Runiq, Sr_avg, left=Sr_avg[0], right=0.0)
    return Sigma_map.reshape(Rmap.shape)

def rotation_curve_from_phi(Rmap, Phi, nbins=80):
    rc, phi_r = radial_profile(Rmap, Phi, nbins=nbins)
    dphi = np.gradient(phi_r, rc, edge_order=2)
    vc2 = np.clip(rc * dphi, a_min=0.0, a_max=None)
    vc = np.sqrt(vc2)
    return rc, vc

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
```

### 12.2 `sparc_loader.py` (robust)
```python
import csv
import numpy as np
from common import G_KPC_KM2_S2_MSUN, moving_average

CANDIDATES = {
    "R_kpc":     ["R_kpc","R","r_kpc","radius_kpc"],
    "Vdisk_kms": ["Vdisk_kms","Vdisk","Vdisk0","Vd"],
    "Vgas_kms":  ["Vgas_kms","Vgas","Vg"],
    "Vbul_kms":  ["Vbul_kms","Vbul","Vb"],
    "Vobs_kms":  ["Vobs_kms","Vobs","V","Vrot"]
}

def _find_col(target, header):
    low = [h.strip().lower() for h in header]
    for cand in CANDIDATES[target]:
        c = cand.lower()
        if c in low:
            return header[low.index(c)]
    return None

def _to_float(s):
    try:
        return float(s)
    except Exception:
        return None

def load_sparc_csv(path, a_disk=0.5, a_bulge=0.7, smooth_w=5):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if len(row) > 0]

    if not rows:
        raise ValueError("CSV ist leer.")

    header = rows[0]; data = rows[1:]

    col_R   = _find_col("R_kpc", header)
    col_Vd  = _find_col("Vdisk_kms", header)
    col_Vg  = _find_col("Vgas_kms", header)
    col_Vb  = _find_col("Vbul_kms", header)
    col_Vo  = _find_col("Vobs_kms", header)

    if not (col_R and col_Vd and col_Vg):
        raise ValueError("Mindestens R_kpc, Vdisk_kms, Vgas_kms müssen vorhanden sein.")

    iR  = header.index(col_R)
    iVd = header.index(col_Vd)
    iVg = header.index(col_Vg)
    iVb = header.index(col_Vb) if col_Vb else None
    iVo = header.index(col_Vo) if col_Vo else None

    R_list = []; Vd_list = []; Vg_list = []; Vb_list = []; Vo_list = []

    for row in data:
        r  = _to_float(row[iR])  if iR  is not None else None
        vd = _to_float(row[iVd]) if iVd is not None else None
        vg = _to_float(row[iVg]) if iVg is not None else None
        if r is None or vd is None or vg is None:
            continue
        vb = _to_float(row[iVb]) if iVb is not None else 0.0
        vo = _to_float(row[iVo]) if iVo is not None else None

        R_list.append(r)
        Vd_list.append(vd)
        Vg_list.append(vg)
        Vb_list.append(vb if vb is not None else 0.0)
        if iVo is not None:
            Vo_list.append(vo if vo is not None else np.nan)

    R    = np.asarray(R_list, dtype=float)
    Vdisk= np.asarray(Vd_list, dtype=float)
    Vgas = np.asarray(Vg_list, dtype=float)
    Vbul = np.asarray(Vb_list, dtype=float) if len(Vb_list)>0 else None
    Vobs = np.asarray(Vo_list, dtype=float) if len(Vo_list)>0 else None

    m = np.isfinite(R) & np.isfinite(Vdisk) & np.isfinite(Vgas)
    if Vbul is not None:
        m &= np.isfinite(Vbul)
    R, Vdisk, Vgas = R[m], Vdisk[m], Vgas[m]
    if Vbul is not None: Vbul = Vbul[m]
    if Vobs is not None:
        pass

    idx = np.argsort(R)
    R = R[idx]; Vdisk = Vdisk[idx]; Vgas = Vgas[idx]
    if Vbul is not None: Vbul = Vbul[idx]
    if Vobs is not None: Vobs = Vobs[idx]

    Vbary2 = (a_disk * Vdisk)**2 + Vgas**2
    if Vbul is not None:
        Vbary2 += (a_bulge * Vbul)**2
    Vbary = np.sqrt(np.clip(Vbary2, 0, None))

    Menc = (R * Vbary**2) / G_KPC_KM2_S2_MSUN
    dM_dR = np.gradient(Menc, R, edge_order=2)
    Sigma_R = (1.0 / (2.0 * np.pi * np.clip(R, 1e-9, None))) * dM_dR

    Sigma_R = moving_average(Sigma_R, w=smooth_w)
    Sigma_R = np.clip(Sigma_R, a_min=0.0, a_max=None)
    if Sigma_R.size >= 2 and not np.isfinite(Sigma_R[0]):
        Sigma_R[0] = Sigma_R[1]

    n = min(R.size, Sigma_R.size)
    R = R[:n]; Sigma_R = Sigma_R[:n]
    if Vobs is not None: Vobs = Vobs[:n]
    if Vobs is not None and not np.any(np.isfinite(Vobs)):
        Vobs = None

    return R, Vobs, Sigma_R
```

### 12.3 `run_ngc3198.py`
```python
# run_ngc3198.py
import os
import numpy as np
from common import (make_grid, radial_to_map,
                    solve_stationary_deficit, solve_screened_poisson, solve_poisson_2d_sigma,
                    rotation_curve_from_phi, quick_plot_maps, quick_plot_rotation)
from sparc_loader import load_sparc_csv

gal_name = "NGC3198"
csv_path = os.path.join("data", "sparc_ngc3198.csv")

nx = ny = 512
dx_kpc = 0.5

ell_kpc = 5.0
tau_gyr = 5.0
D_kpc2_per_gyr = 0.3
Gamma_W = 1.0
chi_const = 0.3

a_disk = 0.5
a_bulge = 0.7

R_sparc, Vobs, Sigma_R = load_sparc_csv(csv_path, a_disk=a_disk, a_bulge=a_bulge, smooth_w=5)

X, Y, Rmap = make_grid(nx, ny, dx_kpc)
Sigma_b = radial_to_map(Rmap, R_sparc, Sigma_R)

nE = Sigma_b / (Sigma_b.max() + 1e-30)
chi = np.full_like(nE, chi_const, dtype=float)

delta = solve_stationary_deficit(nE, chi, Gamma_W, tau_gyr, D_kpc2_per_gyr, dx_kpc)
rho_eff = solve_screened_poisson(delta, ell_kpc, dx_kpc)
Sigma_tot = Sigma_b + rho_eff
Phi = solve_poisson_2d_sigma(Sigma_tot, dx_kpc)

rc, vc_model = rotation_curve_from_phi(Rmap, Phi, nbins=80)

os.makedirs("figures", exist_ok=True)
quick_plot_maps(X, Y, rho_eff, out_png=os.path.join("figures", f"{gal_name}_rho_eff.png"))
rc_obs = R_sparc if Vobs is not None else None
quick_plot_rotation(rc, vc_model, rc_obs=rc_obs, vobs=Vobs,
                    out_png=os.path.join("figures", f"{gal_name}_rotation.png"))
print(f"Done: figures/{gal_name}_rho_eff.png, figures/{gal_name}_rotation.png")
```

### 12.4 `run_ngc2403.py`
```python
# run_ngc2403.py
import os
import numpy as np
from common import (make_grid, radial_to_map,
                    solve_stationary_deficit, solve_screened_poisson, solve_poisson_2d_sigma,
                    rotation_curve_from_phi, quick_plot_maps, quick_plot_rotation)
from sparc_loader import load_sparc_csv

gal_name = "NGC2403"
csv_path = os.path.join("data", "sparc_ngc2403.csv")

nx = ny = 512
dx_kpc = 0.4

ell_kpc = 4.0
tau_gyr = 5.0
D_kpc2_per_gyr = 0.25
Gamma_W = 1.0
chi_const = 0.35

a_disk = 0.5
a_bulge = 0.7

R_sparc, Vobs, Sigma_R = load_sparc_csv(csv_path, a_disk=a_disk, a_bulge=a_bulge, smooth_w=5)

X, Y, Rmap = make_grid(nx, ny, dx_kpc)
Sigma_b = radial_to_map(Rmap, R_sparc, Sigma_R)

nE = Sigma_b / (Sigma_b.max() + 1e-30)
chi = np.full_like(nE, chi_const, dtype=float)

delta = solve_stationary_deficit(nE, chi, Gamma_W, tau_gyr, D_kpc2_per_gyr, dx_kpc)
rho_eff = solve_screened_poisson(delta, ell_kpc, dx_kpc)
Sigma_tot = Sigma_b + rho_eff
Phi = solve_poisson_2d_sigma(Sigma_tot, dx_kpc)

rc, vc_model = rotation_curve_from_phi(Rmap, Phi, nbins=80)

os.makedirs("figures", exist_ok=True)
quick_plot_maps(X, Y, rho_eff, out_png=os.path.join("figures", f"{gal_name}_rho_eff.png"))
rc_obs = R_sparc if Vobs is not None else None
quick_plot_rotation(rc, vc_model, rc_obs=rc_obs, vobs=Vobs,
                    out_png=os.path.join("figures", f"{gal_name}_rotation.png"))
print(f"Done: figures/{gal_name}_rho_eff.png, figures/{gal_name}_rotation.png")
```

### 12.5 `make_sparc_csv.py`
```python
# make_sparc_csv.py
import argparse
import csv
from pathlib import Path

CANONICAL = ["R_kpc", "Vdisk_kms", "Vgas_kms", "Vbul_kms", "Vobs_kms"]

CANDIDATES = {
    "R_kpc":   ["R_kpc","R","r_kpc","radius_kpc"],
    "Vdisk_kms": ["Vdisk_kms","Vdisk","Vdisk0","Vd"],
    "Vgas_kms":  ["Vgas_kms","Vgas","Vg"],
    "Vbul_kms":  ["Vbul_kms","Vbul","Vb"],
    "Vobs_kms":  ["Vobs_kms","Vobs","V","Vrot"],
}

def find_col(target_key, header):
    low = [h.strip().lower() for h in header]
    for cand in CANDIDATES[target_key]:
        c = cand.lower()
        if c in low:
            return header[low.index(c)]
    return None

def write_template_csv(out_path: Path, rmax_kpc: float, dr_kpc: float,
                       include_bulge=True, include_vobs=True, overwrite=False):
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} existiert bereits. --overwrite verwenden.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    radii = []
    r = dr_kpc
    while r <= rmax_kpc + 1e-9:
        radii.append(round(r, 6))
        r += dr_kpc
    header = ["R_kpc", "Vdisk_kms", "Vgas_kms"]
    if include_bulge: header.append("Vbul_kms")
    if include_vobs:  header.append("Vobs_kms")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in radii:
            row = [r, 0.0, 0.0]
            if include_bulge: row.append(0.0)
            if include_vobs:  row.append(0.0)
            w.writerow(row)
    print(f"Template geschrieben: {out_path} (Zeilen: {len(radii)})")
    print(f"Spalten: {', '.join(header)}")

def convert_to_canonical(in_path: Path, out_path: Path, overwrite=False):
    if not in_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {in_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} existiert bereits. --overwrite verwenden.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("Eingabedatei leer.")
    header = rows[0]; data = rows[1:]
    col_R   = find_col("R_kpc", header)
    col_Vd  = find_col("Vdisk_kms", header)
    col_Vg  = find_col("Vgas_kms", header)
    col_Vb  = find_col("Vbul_kms", header)
    col_Vo  = find_col("Vobs_kms", header)
    if not (col_R and col_Vd and col_Vg):
        raise ValueError("Mindestens R_kpc, Vdisk_kms, Vgas_kms müssen gefunden werden.")
    idx = {h: header.index(h) for h in header}
    out_header = ["R_kpc", "Vdisk_kms", "Vgas_kms"]
    if col_Vb: out_header.append("Vbul_kms")
    if col_Vo: out_header.append("Vobs_kms")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(out_header)
        for row in data:
            try:
                r  = float(row[idx[col_R]])
                vd = float(row[idx[col_Vd]])
                vg = float(row[idx[col_Vg]])
            except Exception:
                continue
            out_row = [r, vd, vg]
            if col_Vb:
                try: vb = float(row[idx[col_Vb]])
                except Exception: vb = 0.0
                out_row.append(vb)
            if col_Vo:
                try: vo = float(row[idx[col_Vo]])
                except Exception: vo = 0.0
                out_row.append(vo)
            w.writerow(out_row)
    print(f"Konvertiert → {out_path}")
    print(f"Spalten: {', '.join(out_header)}")

def main():
    import argparse
    p = argparse.ArgumentParser(description="SPARC-CSV-Templates erzeugen oder vorhandene CSVs umbenennen.")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="Leere CSV-Templates erzeugen")
    t.add_argument("--galaxy", required=True)
    t.add_argument("--outdir", default="data")
    t.add_argument("--rmax", type=float, default=35.0)
    t.add_argument("--dr", type=float, default=0.5)
    t.add_argument("--no-bulge", action="store_true")
    t.add_argument("--no-vobs", action="store_true")
    t.add_argument("--overwrite", action="store_true")

    c = sub.add_parser("convert", help="Vorhandene CSV auf kanonische Spalten bringen")
    c.add_argument("--in", dest="inp", required=True)
    c.add_argument("--out", dest="out", required=True)
    c.add_argument("--overwrite", action="store_true")

    args = p.parse_args()
    if args.cmd == "template":
        from pathlib import Path
        gal = args.galaxy.strip().replace(" ", "_").lower()
        out_path = Path(args.outdir) / f"sparc_{gal}.csv"
        write_template_csv(out_path, args.rmax, args.dr,
                           include_bulge=not args.no_bulge,
                           include_vobs=not args.no_vobs,
                           overwrite=args.overwrite)
    elif args.cmd == "convert":
        convert_to_canonical(Path(args.inp), Path(args.out), overwrite=args.overwrite)

if __name__ == "__main__":
    main()
```

---

## 13) Starterpaket (ZIP)
- Vorbereitete Demo-Dateien (synthetische Galaxie): `starter.py`, `config.json`, `README_START.md`.  
- Repo-tauglich; erzeugt `rho_eff_map.png` & `rotation_curve.png`.
- Download (aus dieser Session): **`faden_model_starter.zip`**.

---

## 14) To‑do / Roadmap
- [ ] 1–2 SPARC-Galaxien fitten (Rotation) und Parameter \(\ell,\tau,D,\Gamma_W,\chi\) kalibrieren.  
- [ ] Lensing aus \( \Phi \) ableiten und Profilform prüfen.  
- [ ] CMB/ISW-Plausibilität (frühe/späte Episoden).  
- [ ] Operator-Formalisierung \(\mathcal{O}_W\), Mikro‑Kopplung \(n_E,\chi\) sauber motivieren.  

---

*Erstellt aus der Chat-Zusammenarbeit (MD-Transkript).*
