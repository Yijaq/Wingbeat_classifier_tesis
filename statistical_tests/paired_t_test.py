#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:10:03 2026

@author: YilberQuinto
"""

"""
Paired t-test para comparación de modelos CNN
Aplicado a clasificación de mosquitos con escalogramas wavelet
"""

import numpy as np
from scipy import stats


np.random.seed(42)

# Cada valor representa la accuracy del modelo en ese fold
spectrogram   = np.array([0.8108, 0.8029, 0.8053, 0.8000, 0.8123])

scalogram = np.array([0.8247, 0.8164, 0.8195, 0.8115, 0.8203])

# ─────────────────────────────────────────────
# PAIRED T-TEST
# ─────────────────────────────────────────────

t_stat, p_value = stats.ttest_rel(scalogram, spectrogram )

# ─────────────────────────────────────────────
# TAMAÑO DEL EFECTO: Cohen's d (para muestras pareadas)
# ─────────────────────────────────────────────

differences = scalogram - spectrogram 
cohens_d    = differences.mean() / differences.std(ddof=1)

# ─────────────────────────────────────────────
# INTERVALO DE CONFIANZA AL 95% (Bootstrap)
# ─────────────────────────────────────────────

n_bootstrap = 10_000
bootstrap_diffs = []

for _ in range(n_bootstrap):
    idx     = np.random.choice(len(differences), size=len(differences), replace=True)
    bootstrap_diffs.append(differences[idx].mean())

ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)

# ─────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────

print("=" * 55)
print("   PAIRED T-TEST: scalogram vs spectrogram ")
print("=" * 55)

print(f"\n{'Métrica':<30} {'Valor':>15}")
print("-" * 45)
print(f"{'Media scalogram':<30} {scalogram.mean():>15.4f}")
print(f"{'Media spectrogram':<30} {spectrogram.mean():>15.4f}")
print(f"{'Diferencia media (scalogram- spectrogram)':<30} {differences.mean():>15.5f}")
print("-" * 45)
print(f"{'Estadístico t':<30} {t_stat:>15.4f}")
print(f"{'p-value':<30} {p_value:>15.4e}")
print(f"{'Cohen\'s d':<30} {cohens_d:>15.4f}")
print(f"{'IC 95% Bootstrap [lower]':<30} {ci_lower:>15.5f}")
print(f"{'IC 95% Bootstrap [upper]':<30} {ci_upper:>15.5f}")
print("=" * 55)

# Interpretación automática
alpha = 0.05
print("\n📊 INTERPRETACIÓN:")
if p_value < alpha:
    print(f"  ✅ Diferencia SIGNIFICATIVA (p={p_value:.4e} < α={alpha})")
else:
    print(f"  ❌ Diferencia NO significativa (p={p_value:.4e} ≥ α={alpha})")

if differences.mean() < 0:
    winner = "spectrogram"
else:
    winner = "scalogram"

print(f"  🏆 Modelo superior: {winner}")

# Magnitud del efecto según Cohen
abs_d = abs(cohens_d)
if abs_d < 0.2:
    magnitude = "despreciable"
elif abs_d < 0.5:
    magnitude = "pequeño"
elif abs_d < 0.8:
    magnitude = "mediano"
else:
    magnitude = "grande"

print(f"  📐 Tamaño del efecto: {magnitude} (|d| = {abs_d:.2f})")

if 0 not in range(int(ci_lower * 10000), int(ci_upper * 10000)):
    print(f"IC 95% no contiene 0 → confirma significancia")
