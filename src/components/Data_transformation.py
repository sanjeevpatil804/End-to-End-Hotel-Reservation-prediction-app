import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder,RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

from src.constants.training_pipeline import TARGET_COLUMN, CAT_COLS,NUM_COLS

from src.config.artifact_config import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.config.config import DataTransformationConfig
from src.exception.exception import HotelReservationException

from src.utils.main_utils import save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise HotelReservationException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise HotelReservationException(e, sys)
    def target_encoder(self):
        try:
            label_encoder=LabelEncoder()
            return label_encoder
        except Exception as e:
            raise HotelReservationException(e,sys)

    def get_data_transformer_object(self)->ColumnTransformer:
        
        try:
            num_trans=RobustScaler()
            cat_trans=OrdinalEncoder()

            preprocessing = ColumnTransformer(
                [
                    ('OrdinalEncoder', cat_trans, CAT_COLS),
                    ('RobustScaler', num_trans, NUM_COLS)
                ]
                )
            return preprocessing
        except Exception as e:
            raise HotelReservationException(e,sys)
    
    def apply_smote(self, X_train, y_train):
        """
        Apply SMOTE to handle imbalanced data
        Args:
            X_train: Training features
            y_train: Training target
        Returns:
            X_resampled, y_resampled: Balanced training data
        """
        try:
            smote = SMOTE(random_state=42, sampling_strategy='auto')
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            
            print(f"Original dataset shape: {X_train.shape}")
            print(f"Resampled dataset shape: {X_resampled.shape}")
            print(f"Original class distribution: {np.bincount(y_train.astype(int))}")
            print(f"Resampled class distribution: {np.bincount(y_resampled.astype(int))}")
            
            return X_resampled, y_resampled
        except Exception as e:
            raise HotelReservationException(e,sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        
        try:
        
            train_df=DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            ## training dataframe
            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_en=self.target_encoder()
            target_feature_train_df=target_en.fit_transform(target_feature_train_df)

        
            #testing dataframe
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df=target_en.transform(target_feature_test_df)

            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature=preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)
             
            # Apply SMOTE to handle class imbalance (only on training data)
            transformed_input_train_feature, target_feature_train_df = self.apply_smote(
                transformed_input_train_feature, 
                target_feature_train_df
            )

            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df) ]
            test_arr = np.c_[ transformed_input_test_feature, np.array(target_feature_test_df) ]

            #save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)




            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise HotelReservationException(e,sys)