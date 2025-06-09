import sys
import os
from dataclasses import dataclass
import numpy as np
from src.logger import logging 
import pandas as pd 
from sklearn.compose import ColumnTransformer ### if we want tot create it as pipeline we can use it 
from sklearn.impute import SimpleImputer  # tool that fills in missing values in your dataset.
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exceptions import CustomException
from src.utils import save_object

@dataclass      # @dataclass is a decorator from the dataclasses module (Python 3.7+).
 # It’s used to create classes that mainly hold data — without writing all the boilerplate code like __init__(), __repr__(), etc.
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')  
    #OS.PATH.JOIN():
    # at helps you create file paths in a safe, cross-platform way.It joins one or more strings into a valid file or directory path. It automatically adds the correct slashes (/ or \) based on your operating system

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is resposible for the data transformation 
        
        '''
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=[
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
            ]
            # We create a pipeline 
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('Scaler',StandardScaler())
                ]
        
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
               )
            logging.info("Numerical columns standard scaling is completed ")

            logging.info("categorical columns encoding is completed ")
            preprocessor=ColumnTransformer(
            [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ]
           )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

            
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read Train and test data completed ")

            logging.info("Obtaining the processing object ")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns=['writing_score','reading_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info(
                "Applying the preprocessing object on training daatframe and testing dataframe "
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            #Convert this as np.c_
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info("Saved Preprocessing Object ")


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                            )  #here we are saving the pickle name in the harddisk 
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            