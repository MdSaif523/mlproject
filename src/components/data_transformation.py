import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # Used For handling missing values.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass #-> This decorator 'dataclass' mainly used for assigning the variable .
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
    # This function is responsible for creating all my pickle files which use for converting 'categorical' to 'numerical' features.

        try:
            numerical_columns = ['math_score', 'reading_score', 'writing_score']
            categorical_columns =['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

            num_pipeline=Pipeline( # Pipeline allows you to sequentially apply a list of transformers to preprocess the data.
                
            steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()) # Used for numeric-features.
            ]

            )

            categ_pipeline=Pipeline( #'Pipeline' means ek aisa source jo apne ander se data ko sequentially use krne de.
                
            steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehotEncoder",OneHotEncoder()), # Used for converting categ to numeric-features.
                ("scaler",StandardScaler()) # Used for numeric-features.
            ]

            )

            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorical Columns : {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("numerical_pipeline",num_pipeline,numerical_columns),
                    ("Categorical_pipeline",categ_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
            

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing objects.")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "math_score"

            numerical_columns = [
            "math_score",
            "reading_score",
            "writing_score"
            ]

            input_features_train_df = train_df.drop(
            train_df.drop(columns=[target_column_name])
            )

            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(
            test_df.drop(columns=[target_column_name])
            )

            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocessing objects on training dataframe and testing dataframe.")

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_features_test_df)


            # Add both training input array and training target array.
            train_array=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]


             # Add both test input array and test target array.
            test_array=np.c_[ # 'np.c_'--> Combining with the help of Numpy.
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            ## Saving the pickle file.
            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_array,
                test_array,
            )
        
        except Exception as e:
            raise CustomException(e,sys)

