from src.exception.exception import HotelReservationException   


## configuration of the Data Ingestion Config

from src.config.config import DataIngestionConfig
from src.config.artifact_config import DataIngestionArtifact
import os
import sys
import numpy as np
import pandas as pd
import boto3
from typing import List
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            self.s3_client=boto3.client("s3")
        except Exception as e:
            raise HotelReservationException(e,sys)
        
    def download_from_s3(self):
        """
       download the data from s3 bucket
        """
        try:

            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)

            print(f"Downloading s3://{self.data_ingestion_config.s3_bucket_name}/{self.data_ingestion_config.s3_prefix} ...")
            self.s3_client.download_file(
                Bucket=self.data_ingestion_config.s3_bucket_name,
                Key=self.data_ingestion_config.s3_prefix,
                Filename=self.data_ingestion_config.feature_store_file_path
            )
            df = pd.read_csv(self.data_ingestion_config.feature_store_file_path)
            df.replace({"na": np.nan}, inplace=True)
            return df
        except Exception as e:
            raise HotelReservationException(e,sys)
        
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            
            os.makedirs(dir_path, exist_ok=True)
            
           
            
            train_set.to_csv(
                self.data_ingestion_config.training_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_ingestion_config.testing_file_path, index=False, header=True
            )
            

            
        except Exception as e:
            raise HotelReservationException(e,sys)
        
        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.download_from_s3()
            self.split_data_as_train_test(dataframe)
            dataingestionartifact=DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
                                                        test_file_path=self.data_ingestion_config.testing_file_path)
            return dataingestionartifact

        except Exception as e:
            raise HotelReservationException(e,sys)