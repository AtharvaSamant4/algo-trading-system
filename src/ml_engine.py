# src/ml_engine.py - FINAL CORRECTED VERSION
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib
import logging
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

class MLTradingEngine:
    def __init__(self):
        self.models = {
            'decision_tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=0.1,
                class_weight='balanced'  # Handle class imbalance
            ),
            'random_forest': RandomForestClassifier(
                random_state=42, 
                n_estimators=100,
                max_depth=8,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight='balanced_subsample'  # Handle class imbalance
            )
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        self.feature_names = None
        
    def prepare_features(self, df):
        """Prepare ML features with robust error handling"""
        try:
            # Validate input
            if df is None or df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for features: {len(df) if df is not None else 0} rows")
                return pd.DataFrame()
            
            # Create copy to avoid SettingWithCopyWarning
            df = df.copy()
            features = pd.DataFrame(index=df.index)
            
            # Technical indicators
            features['RSI'] = df.get('RSI', np.nan)
            features['MACD'] = df.get('MACD', 0)
            features['MACD_Signal'] = df.get('MACD_Signal', 0)
            features['MACD_Hist'] = df.get('MACD_Hist', 0)
            features['Volume_Ratio'] = df.get('Volume_Ratio', 1.0)
            features['Volatility'] = df.get('Volatility', 0)
            
            # Price-based features
            features['Price_Change'] = df.get('Price_Change', 0)
            features['Price_Change_5d'] = df.get('Price_Change_5d', 0)
            features['MA_20'] = df.get('MA_20', df['Close'])
            features['MA_50'] = df.get('MA_50', df['Close'])
            
            # MA Ratio with zero division protection
            with np.errstate(divide='ignore', invalid='ignore'):
                ma_ratio = np.where(
                    df['MA_50'] != 0,
                    df['MA_20'] / df['MA_50'],
                    1.0
                )
            features['MA_Ratio'] = np.nan_to_num(ma_ratio, nan=1.0, posinf=1.0, neginf=1.0)
            
            # Momentum features
            features['RSI_MA'] = df['RSI'].rolling(5).mean().fillna(df['RSI'])
            
            # Price position with zero division protection
            rolling_min = df['Close'].rolling(20).min().fillna(df['Close'])
            rolling_max = df['Close'].rolling(20).max().fillna(df['Close'])
            price_range = rolling_max - rolling_min
            price_range = np.where(price_range == 0, 1, price_range)  # Avoid division by zero
            features['Price_Position'] = (df['Close'] - rolling_min) / price_range
            
            # Lag features
            features['RSI_lag1'] = df['RSI'].shift(1).fillna(method='bfill')
            features['Price_Change_lag1'] = df['Price_Change'].shift(1).fillna(0)
            features['Volume_Ratio_lag1'] = df['Volume_Ratio'].shift(1).fillna(1.0)
            
            # Price ratios
            features['Price_MA20_Ratio'] = df['Close'] / df['MA_20']
            features['Price_MA50_Ratio'] = df['Close'] / df['MA_50']
            
            # Handle NaN values - forward fill then backfill
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove any remaining NaNs
            features = features.replace([np.inf, -np.inf], 0)
            features = features.dropna()
            
            logger.info(f"Prepared {len(features)} feature rows")
            return features
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {str(e)}")
            return pd.DataFrame()
    
    def create_target_corrected(self, df, prediction_days=1):
        """Create target without look-ahead bias"""
        try:
            # CORRECTED: Proper next-day return calculation
            # Shift(-1) gets tomorrow's close, today's close is current
            target = ((df['Close'].shift(-prediction_days) > df['Close']).astype(int))
            return target
            
        except Exception as e:
            logger.error(f"Target creation failed: {str(e)}")
            return pd.Series()
    
    def prepare_training_data(self, data_dict):
        """Prepare training data with temporal alignment"""
        all_features = []
        all_targets = []
        
        logger.info("Preparing training data from symbols...")
        
        for symbol, df in data_dict.items():
            logger.info(f"Processing {symbol}...")
            
            # Prepare features
            features = self.prepare_features(df)
            if features.empty:
                logger.warning(f"No features for {symbol}")
                continue
                
            # Create target
            target = self.create_target_corrected(df)
            if target.empty:
                logger.warning(f"No target for {symbol}")
                continue
            
            # Find common indices
            common_index = features.index.intersection(target.index)
            
            if len(common_index) < 50:
                logger.warning(f"Insufficient data for {symbol}: {len(common_index)} rows")
                continue
            
            # CRITICAL FIX: Proper temporal alignment
            # Features at time t predict target at t+1
            aligned_features = features.loc[common_index]
            aligned_target = target.loc[common_index].shift(-1)  # Shift target to next period
            
            # Drop the last row which has no target
            valid_indices = aligned_target.dropna().index
            aligned_features = aligned_features.loc[valid_indices]
            aligned_target = aligned_target.loc[valid_indices]
            
            if len(aligned_features) != len(aligned_target):
                logger.warning(f"Feature-target mismatch for {symbol}: {len(aligned_features)} vs {len(aligned_target)}")
                continue
            
            all_features.append(aligned_features)
            all_targets.append(aligned_target)
            logger.info(f"{symbol}: {len(aligned_features)} samples prepared")
        
        if not all_features:
            raise ValueError("No valid training data prepared")
            
        # Combine data
        X = pd.concat(all_features, axis=0)
        y = pd.concat(all_targets, axis=0)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Combined data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def train_models(self, data_dict):
        """Train ML models with time-series validation"""
        try:
            # Prepare training data
            X, y = self.prepare_training_data(data_dict)
            
            # Handle class imbalance
            class_counts = y.value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            if len(class_counts) < 2:
                logger.warning("Only one class present")
                return {
                    'best_model': 'none',
                    'accuracy': 0.5,
                    'model_scores': {},
                    'feature_importance': {},
                    'classification_report': {}
                }
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            model_scores = {}
            model_metrics = {}
            
            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                
                scores = []
                precisions = []
                recalls = []
                f1s = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                    # Split data
                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Train and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    
                    # Calculate metrics
                    scores.append(accuracy_score(y_val, y_pred))
                    precisions.append(precision_score(y_val, y_pred, average='weighted', zero_division=0))
                    recalls.append(recall_score(y_val, y_pred, average='weighted', zero_division=0))
                    f1s.append(f1_score(y_val, y_pred, average='weighted', zero_division=0))
                
                # Store results
                avg_score = np.mean(scores)
                model_scores[model_name] = avg_score
                model_metrics[model_name] = {
                    'accuracy': avg_score,
                    'precision': np.mean(precisions),
                    'recall': np.mean(recalls),
                    'f1_score': np.mean(f1s),
                    'std': np.std(scores)
                }
                
                logger.info(f"{model_name} CV Accuracy: {avg_score:.4f} ± {np.std(scores):.4f}")
            
            # Select best model
            self.best_model_name = max(model_scores, key=model_scores.get)
            self.best_model = self.models[self.best_model_name]
            
            # Final training
            self.best_model.fit(X_scaled, y)
            
            # Feature importance
            if hasattr(self.best_model, 'feature_importances_'):
                self.feature_importance = dict(zip(self.feature_names, self.best_model.feature_importances_))
                sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"Top features: {sorted_features}")
            
            # Final evaluation
            y_pred = self.best_model.predict(X_scaled)
            final_accuracy = accuracy_score(y, y_pred)
            
            results = {
                'best_model': self.best_model_name,
                'accuracy': final_accuracy,
                'model_scores': model_scores,
                'model_metrics': model_metrics,
                'feature_importance': self.feature_importance,
                'classification_report': classification_report(y, y_pred, output_dict=True, zero_division=0),
                'total_samples': len(X),
                'feature_count': len(self.feature_names)
            }
            
            logger.info(f"Best Model: {self.best_model_name} with accuracy: {final_accuracy:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                'best_model': 'error',
                'accuracy': 0.5,
                'model_scores': {},
                'feature_importance': {},
                'classification_report': {}
            }
    
    def predict_next_day(self, df):
        """Predict next day movement with confidence"""
        try:
            if self.best_model is None:
                logger.warning("Model not trained")
                return 1, 0.5
                
            features = self.prepare_features(df)
            if features.empty:
                logger.warning("No features for prediction")
                return 1, 0.5
                
            # Use latest data
            latest_features = features.iloc[[-1]].copy()
            
            # Handle missing features
            if self.feature_names:
                missing = set(self.feature_names) - set(latest_features.columns)
                for f in missing:
                    latest_features[f] = 0
                latest_features = latest_features[self.feature_names]
            
            # Scale and predict
            features_scaled = self.scaler.transform(latest_features)
            prediction = self.best_model.predict(features_scaled)[0]
            
            # Confidence calculation
            confidence = 0.6  # Default for non-probabilistic models
            if hasattr(self.best_model, 'predict_proba'):
                proba = self.best_model.predict_proba(features_scaled)[0]
                confidence = proba[prediction]
                confidence = max(0.5, min(0.95, confidence))
            
            return int(prediction), float(confidence)
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return 1, 0.5
    
    def save_model(self, filepath):
        """Save trained model"""
        try:
            model_data = {
                'best_model': self.best_model,
                'best_model_name': self.best_model_name,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Model save failed: {str(e)}")
        
    def load_model(self, filepath):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', {})
            self.feature_names = model_data.get('feature_names')
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Model load failed: {str(e)}")