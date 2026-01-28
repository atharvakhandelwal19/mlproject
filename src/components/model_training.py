import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_filepath = os.path.join('artifacts','model_train.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,train_array, test_array):
        try:
            logging.info('Spliting Training and Test Data')

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1]
            )

            models = {
            'KNeighborsRegressor' : KNeighborsRegressor(),
            "Linear Regression": LinearRegression(),
            'AdaBoostRegressor' : AdaBoostRegressor(),
            'DecisionTreeRegressor' : DecisionTreeRegressor(),
            'Random Forest' : RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            'XGBRegressor' : XGBRegressor(),
            'CatBoostRegressor' : CatBoostRegressor(verbose=False)
            }

            # HyperParameter Tuning

            params = {
                "DecisionTreeRegressor" : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter' : ['best', 'random'],
                    'max_features' : ['sqrt', 'log2']
                },

                'Random Forest' : {
                    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features' : ['sqrt', 'log2', None],
                    'n_estimators' : [8,16,32,62,128,256]
                },

                'Gradient Boosting' : {
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8,16,32,64,128,256]
                }, 

                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoostRegressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoostRegressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighborsRegressor": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
                }

            }

            model_report:dict = evaluate_model(X_train=X_train,  y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model = model_report[best_model_name]["model"]
            best_model_score = model_report[best_model_name]["score"]


            if best_model_score < 0.6:
                raise CustomException('No Best Model')
            
            logging.info('Best Model Found')


            save_object(
                file_path= self.model_trainer_config.trained_model_filepath,
                obj= best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
