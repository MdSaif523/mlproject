''' DATA INGESTION STEPS

1. IMPORT DATASETS.
2. CREATE DATAFRAME OF THAT DATA.
3. SAVE WHOLE DATA AS RAW-DATA IN 'Artifacts' FOLDER'.
4. DO TRAIN-TEST-SPLIT.
5. CREATE TEST SET AND TRAIN SET.
6. SAVE TRAIN SET AND TEST SET IN 'Artifacts' FOLDER'.'''

import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "rawdata.csv")


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        try:
            logging.info("Entered the data ingestion method or component")

            # Get project root directory
            project_root = os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
            )

            # Create path to dataset
            data_path = os.path.join(
                project_root,
                "notebook",
                "data",
                "study.csv"
            )

            logging.info(f"Reading dataset from: {data_path}")

            # Read the dataset
            df = pd.read_csv(data_path)

            logging.info("Read the dataset as dataframe")

            # Create artifacts folder if it does not exist
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # Saving training set in artifacts.
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Saving Test set in artifacts.
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_array=data_transformation.initiate_data_transformation(train_data,test_data)

    data_model_trainer=ModelTrainer()
    print(data_model_trainer.initiate_model_trainer(train_array,test_array))
