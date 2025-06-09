import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exceptions import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Splitting Training and Test input Data")
            X_train,Y_train,X_test,Y_test=(
                train_arr[:, :-1], # take out the last column and feed into X_train
                train_arr[:, -1],    #last data as y_train
                test_arr[:, :-1],    #X_test 
                test_arr[:, -1]      #Y_test
            )
            models={
                "Random Forest ":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting ":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours Regressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "AdaBoost Regressor":AdaBoostRegressor()
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=Y_train,X_test=X_test,Y_test=Y_test,models=models)

            #to get the best model score from dictionary 
            best_model_score=max(sorted(model_report.values()))

            # to get the best model from the dictionary 
            best_model_name=list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]
            if best_model_score<0.6:
                raise CustomException("No best model found ")
            logging.info("Best found model on both training and testing dataset ")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2=r2_score(Y_test,predicted)



        except Exception as e:
            raise CustomException(e,sys)