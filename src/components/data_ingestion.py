import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "rawdata.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        try:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            data_path = os.path.join(project_root, "data", "study.csv")
            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)

            data = pd.read_csv(data_path, sep="\t")
            data.to_csv(self.config.raw_data_path, index=False)

            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)

            logging.info("Data ingestion completed.")
            return self.config.train_data_path, self.config.test_data_path
        except Exception as error:
            raise CustomException(error, sys) from error


if __name__ == "__main__":
    train_path, test_path = DataIngestion().initiate_data_ingestion()
    train_array, test_array = DataTransformation().initiate_data_transformation(
        train_path, test_path
    )
    score = ModelTrainer().initiate_model_trainer(train_array, test_array)
    print(f"Model R2 score: {score:.4f}")
