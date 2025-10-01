# sparc_loader.py  (robust)
import csv
import numpy as np
from common import G_KPC_KM2_S2_MSUN, moving_average

# Kandidaten (case-insensitive) für Spaltennamen:
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
    """
    Erwartet mind. R_kpc, Vdisk_kms, Vgas_kms.
    Optional: Vbul_kms, Vobs_kms.
    Gibt zurück:
      R (kpc), Vobs (oder None), Sigma_R (Msun/kpc^2)
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if len(row) > 0]

    if not rows:
        raise ValueError("CSV ist leer.")

    header = rows[0]
    data   = rows[1:]

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

        # Pflichtspalten müssen gültig sein:
        if r is None or vd is None or vg is None:
            continue

        # Optionale Spalten: fehlend/ungültig -> 0 (nicht skippen!)
        vb = _to_float(row[iVb]) if iVb is not None else 0.0
        vo = _to_float(row[iVo]) if iVo is not None else None

        R_list.append(r)
        Vd_list.append(vd)
        Vg_list.append(vg)
        Vb_list.append(vb if vb is not None else 0.0)
        if iVo is not None:
            Vo_list.append(vo if vo is not None else np.nan)

    # Arrays bilden
    R    = np.asarray(R_list, dtype=float)
    Vdisk= np.asarray(Vd_list, dtype=float)
    Vgas = np.asarray(Vg_list, dtype=float)
    Vbul = np.asarray(Vb_list, dtype=float) if len(Vb_list)>0 else None
    Vobs = np.asarray(Vo_list, dtype=float) if len(Vo_list)>0 else None

    # NaNs rauswerfen (konsistent über alle)
    m = np.isfinite(R) & np.isfinite(Vdisk) & np.isfinite(Vgas)
    if Vbul is not None:
        m &= np.isfinite(Vbul)
    R, Vdisk, Vgas = R[m], Vdisk[m], Vgas[m]
    if Vbul is not None: Vbul = Vbul[m]
    if Vobs is not None:
        # Vobs ist optional – sortieren wir separat später; NaNs bleiben erlaubt
        pass

    # Sortieren nach Radius
    idx = np.argsort(R)
    R = R[idx]; Vdisk = Vdisk[idx]; Vgas = Vgas[idx]
    if Vbul is not None: Vbul = Vbul[idx]
    if Vobs is not None: Vobs = Vobs[idx]  # NaNs bleiben am Platz

    # Baryonischer Beitrag (stellare Komp. skaliert)
    Vbary2 = (a_disk * Vdisk)**2 + Vgas**2
    if Vbul is not None:
        Vbary2 += (a_bulge * Vbul)**2
    Vbary = np.sqrt(np.clip(Vbary2, 0, None))

    # M(<R) und abgeleitete Σ(R)
    Menc = (R * Vbary**2) / G_KPC_KM2_S2_MSUN  # Msun
    dM_dR = np.gradient(Menc, R, edge_order=2)
    Sigma_R = (1.0 / (2.0 * np.pi * np.clip(R, 1e-9, None))) * dM_dR  # Msun/kpc^2

    # Glätten, Clip
    Sigma_R = moving_average(Sigma_R, w=smooth_w)
    Sigma_R = np.clip(Sigma_R, a_min=0.0, a_max=None)

    # R=0-Fall: setze ersten Wert auf zweiten
    if Sigma_R.size >= 2 and not np.isfinite(Sigma_R[0]):
        Sigma_R[0] = Sigma_R[1]

    # Sicherheit: gleiche Länge garantieren
    n = min(R.size, Sigma_R.size)
    R = R[:n]; Sigma_R = Sigma_R[:n]
    if Vobs is not None: Vobs = Vobs[:n]

    # Falls Vobs komplett NaN → None zurückgeben
    if Vobs is not None and not np.any(np.isfinite(Vobs)):
        Vobs = None

    return R, Vobs, Sigma_R
