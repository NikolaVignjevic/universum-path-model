# universum-path-model

**Kurz:** Numerisches Minimalmodell für ein Stoßwellen-/Faden-Universum.  
Die Stoßwelle erzeugt Skalarfeld-Defizite (δρ_φ), daraus entsteht eine effektive Dichte ρ_eff (DM-ähnliche Spur), die zusammen mit baryonischer Materie das Potential Φ bestimmt. Ziel: Rotationskurven und Lensing konsistent reproduzieren, **EEP-konform**.

## Postulate (P1–P5)
- **P1** Bulk = superfluides skalares Medium; Stoßwelle `W` reorganisiert es.  
- **P2** Fäden/Knoten entstehen durch Entnahme aus dem Skalarfeld.  
- **P3** Defizite erzeugen ein Potential Φ, an das alle Sonden gleich koppeln (EEP).  
- **P4** Defizite relaxen/diffundieren langsam → persistente **Spuren** (DM-artig).  
- **P5** Kohärenzregel: Kohärente Materie (BEC-artig) trägt normal zu Masse bei, erzeugt aber **gedämpfte** Spur (kleiner zusätzlicher ρ_eff-Beitrag).

## Minimalmodell (praxisnah)
Stationär:
- Defizit:  (1/τ)·δρ_φ − D·∇²δρ_φ = Γ_W·(1−χ)·n_E  
- Spur:     ρ_eff − ℓ²·∇²ρ_eff = δρ_φ  
- Potential: ∇²Φ = 4πG·(ρ_mass + ρ_eff)

Parameter: ℓ (kpc), τ (Gyr), D (kpc²/Gyr), Γ_W (Norm), χ∈[0,1].

## Quickstart
Voraussetzungen: Python 3.10+, `numpy`, `matplotlib`.

```bash
pip install numpy matplotlib
python starter.py
```
---
