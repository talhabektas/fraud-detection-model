"""
Machine Learning Model Training for Fraud Detection
Trains Random Forest and XGBoost models with hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    precision_recall_curve,
    roc_curve,
    f1_score
)
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Import preprocessing module
from preprocessing import preprocess_pipeline, FraudPreprocessor


class FraudDetectionModel:
    """Fraud Detection Model Trainer"""
    
    def __init__(self, model_type='random_forest'):
        """
        Initialize model
        
        Args:
            model_type: 'random_forest' or 'xgboost'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        
    def create_model(self):
        """Create model instance"""
        if self.model_type == 'random_forest':
            print("🌲 Creating Random Forest Classifier...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif self.model_type == 'xgboost':
            print("🚀 Creating XGBoost Classifier...")
            # Calculate scale_pos_weight for imbalanced data
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        print("\n" + "=" * 70)
        print(f"🎯 TRAINING {self.model_type.upper()} MODEL")
        print("=" * 70)
        
        if self.model is None:
            self.create_model()
        
        print(f"\n📊 Training data shape: {X_train.shape}")
        print(f"   Class distribution: Fraud={y_train.sum():,}, Normal={(y_train==0).sum():,}")
        
        # Train model
        if self.model_type == 'xgboost' and X_val is not None:
            print("\n🏋️  Training with early stopping...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            print("\n🏋️  Training model...")
            self.model.fit(X_train, y_train)
        
        print("\n✅ Training complete!")
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
    
    def evaluate(self, X_test, y_test, dataset_name='Test'):
        """Evaluate model performance"""
        print("\n" + "=" * 70)
        print(f"📊 EVALUATING ON {dataset_name.upper()} SET")
        print("=" * 70)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        print("\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n🔢 Confusion Matrix:")
        print(f"   True Negatives:  {cm[0,0]:,}")
        print(f"   False Positives: {cm[0,1]:,}")
        print(f"   False Negatives: {cm[1,0]:,}")
        print(f"   True Positives:  {cm[1,1]:,}")
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")
        
        # F1 Score
        f1 = f1_score(y_test, y_pred)
        print(f"📊 F1 Score: {f1:.4f}")
        
        return {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'f1_score': f1
        }
    
    def plot_feature_importance(self, feature_names, top_n=20, save_path=None):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("⚠️  No feature importance available")
            return
        
        print(f"\n📊 Plotting top {top_n} important features...")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type.upper()}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Saved plot to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return importance_df
    
    def save_model(self, save_dir='src/ml_model'):
        """Save trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(save_dir, f'{self.model_type}_model.pkl')
        latest_path = os.path.join(save_dir, 'model.pkl')
        
        # Save model
        joblib.dump(self.model, model_path)
        joblib.dump(self.model, latest_path)  # Also save as latest
        
        print(f"\n💾 Model saved:")
        print(f"   - {model_path}")
        print(f"   - {latest_path} (latest)")
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = os.path.join(save_dir, 'feature_importance.npy')
            np.save(importance_path, self.feature_importance)
            print(f"   - {importance_path}")
    
    @staticmethod
    def load_model(model_path='src/ml_model/model.pkl'):
        """Load a saved model"""
        model = joblib.load(model_path)
        print(f"📂 Loaded model from {model_path}")
        return model


def train_fraud_detection_model(
    data_path='data/creditcard.csv',
    model_type='random_forest',
    use_smote=True
):
    """
    Complete training pipeline
    
    Args:
        data_path: Path to dataset
        model_type: 'random_forest' or 'xgboost'
        use_smote: Whether to use SMOTE for class balancing
    """
    print("\n" + "=" * 70)
    print("🚀 FRAUD DETECTION MODEL - TRAINING PIPELINE")
    print("=" * 70)
    print(f"   Model Type: {model_type.upper()}")
    print(f"   SMOTE: {'Enabled' if use_smote else 'Disabled'}")
    print("=" * 70 + "\n")
    
    # 1. Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = preprocess_pipeline(
        data_path, 
        use_smote=use_smote,
        save_preprocessor=True
    )
    
    # 2. Train model
    trainer = FraudDetectionModel(model_type=model_type)
    trainer.train(X_train, y_train, X_val, y_val)
    
    # 3. Evaluate on validation set
    val_metrics = trainer.evaluate(X_val, y_val, dataset_name='Validation')
    
    # 4. Evaluate on test set
    test_metrics = trainer.evaluate(X_test, y_test, dataset_name='Test')
    
    # 5. Plot feature importance
    importance_df = trainer.plot_feature_importance(
        feature_cols, 
        top_n=20,
        save_path=f'src/ml_model/feature_importance_{model_type}.png'
    )
    
    # 6. Save model
    trainer.save_model()
    
    print("\n" + "=" * 70)
    print("✅ TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 Final Test Set Performance:")
    print(f"   ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   F1 Score: {test_metrics['f1_score']:.4f}")
    print("=" * 70 + "\n")
    
    return trainer, test_metrics


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Fraud Detection Model')
    parser.add_argument('--model', type=str, default='random_forest', 
                       choices=['random_forest', 'xgboost'],
                       help='Model type to train')
    parser.add_argument('--no-smote', action='store_true',
                       help='Disable SMOTE')
    parser.add_argument('--data', type=str, default='data/creditcard.csv',
                       help='Path to dataset')
    
    args = parser.parse_args()
    
    # Train model
    trainer, metrics = train_fraud_detection_model(
        data_path=args.data,
        model_type=args.model,
        use_smote=not args.no_smote
    )
