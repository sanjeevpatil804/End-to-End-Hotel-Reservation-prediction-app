import sys
import optuna
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from src.exception.exception import HotelReservationException


def optimize_xgboost(X_train, y_train, n_trials=50):
    """
    Optimize XGBoost hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training target
        n_trials: Number of optimization trials
        
    Returns:
        dict: Best hyperparameters found
    """
    try:
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
            
            model = XGBClassifier(**params)
            score = cross_val_score(model, X_train, y_train, cv=3, 
                                   scoring='f1', n_jobs=1).mean()
            return score
        
        # Run Optuna optimization
        print("Starting Optuna hyperparameter tuning...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest F1 Score: {study.best_value:.4f}")
        print(f"Best Parameters: {study.best_params}")
        
        return study.best_params
        
    except Exception as e:
        raise HotelReservationException(e, sys)
