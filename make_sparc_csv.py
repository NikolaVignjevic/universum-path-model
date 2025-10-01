# make_sparc_csv.py
# Hilfswerkzeug für universum-path-model
# - Erstellt Template-CSV-Dateien mit den erwarteten Spalten:
#   R_kpc, Vdisk_kms, Vgas_kms, (optional) Vbul_kms, Vobs_kms
# - Konvertiert vorhandene SPARC-CSV-Dateien auf diese Spaltennamen.

import argparse
import csv
from pathlib import Path

# ---- Kanonische Spaltennamen, die unser Loader erwartet ----
CANONICAL = ["R_kpc", "Vdisk_kms", "Vgas_kms", "Vbul_kms", "Vobs_kms"]

# ---- Kandidatennamen (Groß/Kleinschreibung egal) für Auto-Erkennung ----
CANDIDATES = {
    "R_kpc":   ["R_kpc","R","r_kpc","radius_kpc"],
    "Vdisk_kms": ["Vdisk_kms","Vdisk","Vdisk0","Vd"],
    "Vgas_kms":  ["Vgas_kms","Vgas","Vg"],
    "Vbul_kms":  ["Vbul_kms","Vbul","Vb"],
    "Vobs_kms":  ["Vobs_kms","Vobs","V","Vrot"],
}

def find_col(target_key, header):
    """Finde in header eine Spalte, die zu target_key passt (kandidatenbasiert)."""
    low = [h.strip().lower() for h in header]
    for cand in CANDIDATES[target_key]:
        c = cand.lower()
        if c in low:
            return header[low.index(c)]
    return None

def write_template_csv(out_path: Path, rmax_kpc: float, dr_kpc: float,
                       include_bulge=True, include_vobs=True, overwrite=False):
    """Erzeuge leere Template-CSV mit kanonischen Spalten und Radiusschritten."""
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} existiert bereits. --overwrite verwenden oder anderen Pfad wählen.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Radii: von dr bis rmax in Schritten dr
    radii = []
    r = dr_kpc
    while r <= rmax_kpc + 1e-9:
        radii.append(round(r, 6))
        r += dr_kpc

    # Header zusammenbauen
    header = ["R_kpc", "Vdisk_kms", "Vgas_kms"]
    if include_bulge:
        header.append("Vbul_kms")
    if include_vobs:
        header.append("Vobs_kms")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        # Fülle mit Nullen als Platzhalter (die kannst du später überschreiben)
        for r in radii:
            row = [r, 0.0, 0.0]
            if include_bulge:
                row.append(0.0)
            if include_vobs:
                row.append(0.0)
            w.writerow(row)

    print(f"Template geschrieben: {out_path}  (Zeilen: {len(radii)})")
    print(f"Spalten: {', '.join(header)}")


def convert_to_canonical(in_path: Path, out_path: Path, overwrite=False):
    """
    Lies eine bestehende CSV (SPARC o.ä.), erkenne Spalten automatisch
    und schreibe eine neue CSV mit kanonischen Spaltennamen.
    Fehlende optionale Spalten (Bulge/Obs) werden weggelassen.
    """
    if not in_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {in_path}")
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} existiert bereits. --overwrite verwenden oder anderen Pfad wählen.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        raise ValueError("Eingabedatei leer.")

    header = rows[0]
    data   = rows[1:]

    # Spalten in Eingabe finden
    col_R   = find_col("R_kpc", header)
    col_Vd  = find_col("Vdisk_kms", header)
    col_Vg  = find_col("Vgas_kms", header)
    col_Vb  = find_col("Vbul_kms", header)
    col_Vo  = find_col("Vobs_kms", header)

    if not (col_R and col_Vd and col_Vg):
        raise ValueError("Mindestens R_kpc, Vdisk_kms, Vgas_kms müssen gefunden werden.")

    idx = {h: header.index(h) for h in header}

    # Zielheader
    out_header = ["R_kpc", "Vdisk_kms", "Vgas_kms"]
    if col_Vb:
        out_header.append("Vbul_kms")
    if col_Vo:
        out_header.append("Vobs_kms")

    # Schreiben
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out_header)
        for row in data:
            try:
                r  = float(row[idx[col_R]])
                vd = float(row[idx[col_Vd]])
                vg = float(row[idx[col_Vg]])
            except Exception:
                continue  # überspringe nichtnumerische Zeilen

            out_row = [r, vd, vg]
            if col_Vb:
                try:
                    vb = float(row[idx[col_Vb]])
                except Exception:
                    vb = 0.0
                out_row.append(vb)
            if col_Vo:
                try:
                    vo = float(row[idx[col_Vo]])
                except Exception:
                    vo = 0.0
                out_row.append(vo)
            w.writerow(out_row)

    print(f"Konvertiert → {out_path}")
    print(f"Spalten: {', '.join(out_header)}")


def main():
    p = argparse.ArgumentParser(description="SPARC-CSV-Templates erzeugen oder vorhandene CSVs umbenennen.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Subcommand: template
    t = sub.add_parser("template", help="Leere CSV-Templates erzeugen")
    t.add_argument("--galaxy", required=True, help="Name (für Dateinamen)")
    t.add_argument("--outdir", default="data", help="Zielordner (default: data)")
    t.add_argument("--rmax", type=float, default=35.0, help="Max. Radius in kpc (default: 35)")
    t.add_argument("--dr", type=float, default=0.5, help="Schrittweite in kpc (default: 0.5)")
    t.add_argument("--no-bulge", action="store_true", help="Bulge-Spalte weglassen")
    t.add_argument("--no-vobs", action="store_true", help="Vobs-Spalte weglassen")
    t.add_argument("--overwrite", action="store_true", help="Existierende Datei überschreiben")

    # Subcommand: convert
    c = sub.add_parser("convert", help="Vorhandene CSV auf kanonische Spalten bringen")
    c.add_argument("--in", dest="inp", required=True, help="Eingabe-CSV")
    c.add_argument("--out", dest="out", required=True, help="Ausgabe-CSV")
    c.add_argument("--overwrite", action="store_true", help="Existierende Datei überschreiben")

    args = p.parse_args()

    if args.cmd == "template":
        gal = args.galaxy.strip().replace(" ", "_").lower()
        outdir = Path(args.outdir)
        out_path = outdir / f"sparc_{gal}.csv"
        write_template_csv(
            out_path=out_path,
            rmax_kpc=args.rmax,
            dr_kpc=args.dr,
            include_bulge=not args.no_bulge,
            include_vobs=not args.no_vobs,
            overwrite=args.overwrite
        )
        # Für Komfort: NGC-spezifische Defaults empfehlen
        if "3198" in gal:
            print("Hinweis: Für NGC 3198 sind rmax≈40 kpc, dr≈0.5 kpc brauchbare Startwerte.")
        if "2403" in gal:
            print("Hinweis: Für NGC 2403 sind rmax≈25 kpc, dr≈0.4–0.5 kpc brauchbare Startwerte.")

    elif args.cmd == "convert":
        convert_to_canonical(
            in_path=Path(args.inp),
            out_path=Path(args.out),
            overwrite=args.overwrite
        )

if __name__ == "__main__":
    main()
