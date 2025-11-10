import os
import sys

from src.exception.exception import HotelReservationException
from src.config.artifact_config import DataTransformationArtifact, ModelTrainerArtifact
from src.config.config import ModelTrainerConfig
from src.utils.estimator import NetworkModel
from src.utils.main_utils import save_object, load_object, load_numpy_array_data
from src.utils.classification_metrics import get_classification_score
from src.utils.optuna_tuner import optimize_xgboost
from xgboost import XGBClassifier





class ModelTrainer:
    """
    Model trainer class for training and evaluating machine learning models
    """
    
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize ModelTrainer with configuration and data transformation artifacts
        
        Args:
            model_trainer_config: Configuration for model training
            data_transformation_artifact: Artifact containing transformed data paths
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise HotelReservationException(e, sys)
        
        
    def train_model(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate XGBoost model with Optuna hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            ModelTrainerArtifact: Artifact containing trained model and metrics
        """
        try:
            # Optimize hyperparameters using Optuna (only on training data)
            best_params = optimize_xgboost(X_train, y_train, n_trials=10)
            
            # Train final model with best parameters
            print("\nTraining final model with best parameters...")
            model = XGBClassifier(**best_params, random_state=42, n_jobs=-1, verbosity=1)
            model.fit(X_train, y_train)
            
            # Evaluate on training set
            print("Evaluating on training set...")
            y_train_pred = model.predict(X_train)
            classification_train_metric = get_classification_score(
                y_true=y_train, 
                y_pred=y_train_pred
            )
            
            # Evaluate on test set
            print("Evaluating on test set...")
            y_test_pred = model.predict(X_test)
            classification_test_metric = get_classification_score(
                y_true=y_test, 
                y_pred=y_test_pred
            )
            print(f"Training F1 Score: {classification_train_metric.f1_score}")
            print(f"Test F1 Score: {classification_test_metric.f1_score}")  
            
            # Load preprocessor
            preprocessor = load_object(
                file_path=self.data_transformation_artifact.transformed_object_file_path
            )
            
            # Create model directory
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # Create and save network model
            network_model = NetworkModel(preprocessor=preprocessor, model=model)
            save_object(
                self.model_trainer_config.trained_model_file_path, 
                obj=network_model
            )
            
            # Save final model separately
            final_model_dir = "final_model"
            os.makedirs(final_model_dir, exist_ok=True)
            save_object(os.path.join(final_model_dir, "model.pkl"), model)
            
            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            return model_trainer_artifact
            
        except Exception as e:
            raise HotelReservationException(e, sys)

        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiate model training process
        
        Returns:
            ModelTrainerArtifact: Artifact containing trained model and metrics
        """
        try:
            # Get file paths
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            # Load training and testing arrays
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            # Split features and target
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            # Train model
            model_trainer_artifact = self.train_model(X_train, y_train, X_test, y_test)
            return model_trainer_artifact
            
        except Exception as e:
            raise HotelReservationException(e, sys)
            
