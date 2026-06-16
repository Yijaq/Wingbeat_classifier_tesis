#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:07:36 2026

@author: YilberQuinto
"""

import numpy as np
import scipy.stats as stats

def calcular_ic_folds(rendimientos, confianza=0.95):
    """Calcula el IC de la media a partir de un array de resultados de los folds."""
    datos = np.array(rendimientos)
    n = len(datos)
    media = np.mean(datos)
    
    # Error estándar de la media (SEM)
    sem = stats.sem(datos) 
    
    # Intervalo usando la distribución t de Student (df = grados de libertad)
    intervalo = stats.t.interval(confianza, df=n-1, loc=media, scale=sem)
    
    return media, intervalo

# Precisiones obtenidas en 5 folds para el Escalograma
hd_scalogram_vgg13_folds = [0.8317, 0.8378, 0.8457, 0.8316, 0.8359]
hd_scalogram_vgg16_folds = [0.8358, 0.8363, 0.8360, 0.8348, 0.8278]
hd_scalogram_vgg19_folds = [0.8319, 0.8273, 0.8408, 0.8296, 0.8213]
hd_scalogram_xcep_folds = [0.8248, 0.8236, 0.8360, 0.8223, 0.8232]
hd_scalogram_resnet50_folds = [0.8172, 0.8132, 0.8271, 0.8146, 0.8250]

scalogram_vgg13_folds = [0.8015, 0.8007, 0.8123, 0.8007, 0.7981]
scalogram_vgg16_folds = [0.7994, 0.8014, 0.8063, 0.7985, 0.7953]
scalogram_vgg19_folds = [0.7923, 0.7956, 0.8064, 0.7914, 0.7904]
scalogram_xcep_folds = [0.8011, 0.8103, 0.8166, 0.8041, 0.7991]
scalogram_resnet50_folds = [0.7963, 0.7946, 0.7986, 0.7943, 0.7912]

spectrogram_vgg13_folds = [0.8193, 0.8137, 0.8252, 0.8097, 0.8229]
spectrogram_vgg16_folds = [0.8135, 0.8096, 0.8134, 0.8184, 0.8183]
spectrogram_vgg19_folds = [0.8167, 0.8103, 0.8125, 0.8055, 0.8084]
spectrogram_xcep_folds = [0.8034, 0.8047, 0.8088, 0.8053, 0.8124]
spectrogram_resnet50_folds = [0.7681, 0.7725, 0.7646, 0.7608, 0.7862]

mel_spectrogram_vgg13_folds = [0.8247, 0.8164, 0.8195, 0.8115, 0.8203]
mel_spectrogram_vgg16_folds = [0.8178, 0.8108, 0.8194, 0.8105, 0.8212]
mel_spectrogram_vgg19_folds = [0.8152, 0.8023, 0.8111, 0.8016, 0.8156]
mel_spectrogram_xcep_folds = [0.8113, 0.8019, 0.8085, 0.8046, 0.8130]
mel_spectrogram_resnet50_folds = [0.7721, 0.7906, 0.7876, 0.7928, 0.7917]

mfcc_vgg13_folds = [0.8108, 0.8029, 0.8053, 0.8000, 0.8123]
mfcc_vgg16_folds = [0.8094, 0.7989, 0.8071, 0.7932, 0.7999]
mfcc_vgg19_folds = [0.8001, 0.7991, 0.8025, 0.7892, 0.8049]
mfcc_xcep_folds = [0.7938, 0.7979, 0.7935, 0.7868, 0.8016]
mfcc_resnet50_folds = [0.7697, 0.7697, 0.7643, 0.7606, 0.7755]


media_acc, (ic_inf, ic_sup) = calcular_ic_folds(mel_spectrogram_resnet50_folds)

print("--- Resultados de Validación Cruzada (5 Folds) ---")
print(f"Precisión Media: {media_acc*100:.2f}%")
print(f"IC 95% de la Media: [{ic_inf*100:.2f}, {ic_sup*100:.2f}]")