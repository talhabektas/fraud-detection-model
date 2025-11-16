"""
Data Preprocessing Module for Fraud Detection
Handles feature engineering, scaling, and SMOTE for class imbalance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os


class FraudPreprocessor:
    """Preprocessor for credit card fraud detection data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42, sampling_strategy=0.5)
        
    def load_data(self, filepath):
        """Load credit card transaction data"""
        print(f"📊 Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df):,} transactions")
        print(f"   - Fraud cases: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
        print(f"   - Normal cases: {(df['Class']==0).sum():,}")
        return df
    
    def feature_engineering(self, df):
        """Create additional features"""
        print("\n🔧 Feature Engineering...")
        
        # Time-based features
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Day'] = (df['Time'] // (3600 * 24))
        
        # Amount-based features
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Interaction features
        df['V1_Amount'] = df['V1'] * df['Amount']
        df['V2_Amount'] = df['V2'] * df['Amount']
        
        print("✅ Created time-based and interaction features")
        return df
    
    def prepare_features(self, df, is_training=True):
        """Prepare features for modeling"""
        print("\n⚙️  Preparing features...")
        
        # Define feature columns
        feature_cols = [col for col in df.columns if col not in ['Class', 'Time']]
        
        X = df[feature_cols].copy()
        y = df['Class'].copy() if 'Class' in df.columns else None
        
        # Scale features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            print(f"✅ Fitted scaler on {len(feature_cols)} features")
        else:
            X_scaled = self.scaler.transform(X)
            print(f"✅ Scaled features using fitted scaler")
        
        # Convert back to DataFrame
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        return X_scaled, y, feature_cols
    
    def handle_imbalance(self, X, y):
        """Apply SMOTE to handle class imbalance"""
        print("\n⚖️  Handling class imbalance with SMOTE...")
        print(f"   Before SMOTE - Class distribution:")
        print(f"     Normal: {(y==0).sum():,}")
        print(f"     Fraud:  {(y==1).sum():,}")
        
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        print(f"   After SMOTE - Class distribution:")
        print(f"     Normal: {(y_resampled==0).sum():,}")
        print(f"     Fraud:  {(y_resampled==1).sum():,}")
        
        return X_resampled, y_resampled
    
    def split_data(self, X, y, test_size=0.2, validation_size=0.1):
        """Split data into train, validation, and test sets"""
        print(f"\n✂️  Splitting data (test={test_size}, val={validation_size})...")
        
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train and validation
        val_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"   Train set:      {len(X_train):,} samples")
        print(f"   Validation set: {len(X_val):,} samples")
        print(f"   Test set:       {len(X_test):,} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessor(self, save_dir='src/ml_model'):
        """Save the fitted preprocessor"""
        os.makedirs(save_dir, exist_ok=True)
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"\n💾 Saved scaler to {scaler_path}")
    
    @staticmethod
    def load_preprocessor(load_dir='src/ml_model'):
        """Load a saved preprocessor"""
        preprocessor = FraudPreprocessor()
        scaler_path = os.path.join(load_dir, 'scaler.pkl')
        preprocessor.scaler = joblib.load(scaler_path)
        print(f"📂 Loaded scaler from {scaler_path}")
        return preprocessor


def preprocess_pipeline(data_path, use_smote=True, save_preprocessor=True):
    """
    Complete preprocessing pipeline
    
    Args:
        data_path: Path to creditcard.csv
        use_smote: Whether to apply SMOTE
        save_preprocessor: Whether to save the fitted preprocessor
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols)
    """
    print("=" * 70)
    print("🚀 FRAUD DETECTION - DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    preprocessor = FraudPreprocessor()
    
    # 1. Load data
    df = preprocessor.load_data(data_path)
    
    # 2. Feature engineering
    df = preprocessor.feature_engineering(df)
    
    # 3. Prepare features
    X, y, feature_cols = preprocessor.prepare_features(df, is_training=True)
    
    # 4. Split data BEFORE SMOTE (to avoid data leakage)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # 5. Apply SMOTE only on training data
    if use_smote:
        X_train, y_train = preprocessor.handle_imbalance(X_train, y_train)
    
    # 6. Save preprocessor
    if save_preprocessor:
        preprocessor.save_preprocessor()
    
    print("\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70 + "\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


if __name__ == "__main__":
    # Test preprocessing
    data_path = "data/creditcard.csv"
    X_train, X_val, X_test, y_train, y_val, y_test, features = preprocess_pipeline(data_path)
    
    print(f"\n📊 Final Data Shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   X_val:   {X_val.shape}")
    print(f"   X_test:  {X_test.shape}")
    print(f"   Features: {len(features)}")
