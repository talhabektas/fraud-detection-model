#!/usr/bin/env python3
"""
Model dosyalarını incele
"""

import pickle
import numpy as np
import os

def inspect_model_files():
    """Model ve scaler dosyalarını incele"""
    
    model_dir = "src/ml_model"
    
    # Random Forest Model dosyası
    model_path = f"{model_dir}/random_forest_model.pkl"
    if os.path.exists(model_path):
        print("=" * 60)
        print("🤖 RANDOM FOREST MODEL DOSYASI")
        print("=" * 60)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model Tipi: {type(model).__name__}")
        print(f"Model Sınıfı: {model.__class__.__name__}")
        
        if hasattr(model, 'n_estimators'):
            print(f"Ağaç Sayısı: {model.n_estimators}")
        if hasattr(model, 'max_depth'):
            print(f"Max Derinlik: {model.max_depth}")
        if hasattr(model, 'n_features_in_'):
            print(f"Feature Sayısı: {model.n_features_in_}")
        if hasattr(model, 'classes_'):
            print(f"Sınıflar: {model.classes_}")
        
        # Dosya boyutu
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Dosya Boyutu: {size_mb:.2f} MB")
    else:
        print(f"❌ Model dosyası bulunamadı: {model_path}")
    
    print("\n")
    
    # Scaler dosyası
    scaler_path = f"{model_dir}/scaler.pkl"
    if os.path.exists(scaler_path):
        print("=" * 60)
        print("📊 SCALER DOSYASI")
        print("=" * 60)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print(f"Scaler Tipi: {type(scaler).__name__}")
        
        if hasattr(scaler, 'mean_'):
            print(f"Mean değerleri: {scaler.mean_[:5]}... (ilk 5)")
        if hasattr(scaler, 'scale_'):
            print(f"Scale değerleri: {scaler.scale_[:5]}... (ilk 5)")
        if hasattr(scaler, 'n_features_in_'):
            print(f"Feature Sayısı: {scaler.n_features_in_}")
        
        # Dosya boyutu
        size_kb = os.path.getsize(scaler_path) / 1024
        print(f"Dosya Boyutu: {size_kb:.2f} KB")
    else:
        print(f"❌ Scaler dosyası bulunamadı: {scaler_path}")
    
    print("\n")
    
    # NPY dosyaları (varsa)
    npy_files = [f for f in os.listdir(model_dir) if f.endswith('.npy')]
    if npy_files:
        print("=" * 60)
        print("📁 NPY DOSYALARI")
        print("=" * 60)
        for npy_file in npy_files:
            npy_path = f"{model_dir}/{npy_file}"
            data = np.load(npy_path)
            print(f"\n{npy_file}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Min: {data.min():.4f}, Max: {data.max():.4f}")
            size_kb = os.path.getsize(npy_path) / 1024
            print(f"  Dosya Boyutu: {size_kb:.2f} KB")

if __name__ == "__main__":
    inspect_model_files()
