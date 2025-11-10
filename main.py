from src.components.Data_transformation import DataTransformation
from src.components.data_ingetion import DataIngestion
from src.components.Data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.config.config import DataTransformationConfig, ModelTrainerConfig, TrainingPipelineConfig,DataIngestionConfig, DataValidationConfig
from src.config.artifact_config import DataIngestionArtifact
import sys

from src.exception.exception import HotelReservationException
    



if __name__=='__main__':
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        dataingestionartifact=data_ingestion.initiate_data_ingestion()

        data_validation_config= DataValidationConfig(trainingpipelineconfig)
        data_validation= DataValidation(dataingestionartifact, data_validation_config)
        datavalidationartifact= data_validation.initiate_data_validation()


        data_transformation_config=DataTransformationConfig(trainingpipelineconfig)
        data_transformation=DataTransformation(datavalidationartifact,data_transformation_config)
        data_transformation_artifact=data_transformation.initiate_data_transformation()

        model_trainer_config=ModelTrainerConfig(trainingpipelineconfig)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact=model_trainer.initiate_model_trainer()




    except Exception as e:
        raise HotelReservationException(e,sys)
