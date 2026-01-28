import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from tqdm import tqdm
import optuna
from sklearn.base import clone
from catboost import CatBoostRegressor

def save_object(file_path, obj):
    try: 
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in tqdm(models.items(), desc="Training Models"):
            param_grid = params.get(model_name, {})

            if not param_grid:
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
                report[model_name] = {"model": model, "score": score}
                continue

            if model_name == "CatBoostRegressor":
                def objective(trial):
                    trial_params = {
                        "depth": trial.suggest_int("depth", 6, 10),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                        "iterations": trial.suggest_int("iterations", 30, 100),
                        "verbose": False
                    }

                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    scores = []

                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]

                        cb_model = CatBoostRegressor(**trial_params)
                        cb_model.fit(X_tr, y_tr)
                        preds = cb_model.predict(X_val)

                        scores.append(r2_score(y_val, preds))

                    return np.mean(scores)

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=30, show_progress_bar=False)

                best_model = CatBoostRegressor(
                    **study.best_params,
                    verbose=False
                )
                best_model.fit(X_train, y_train)

                test_score = r2_score(y_test, best_model.predict(X_test))

                report[model_name] = {
                    "model": best_model,
                    "score": test_score
                }
                continue

            def objective(trial):
                trial_params = {}
                for param_name, values in param_grid.items():
                    trial_params[param_name] = trial.suggest_categorical(
                        param_name, values
                    )

                model_clone = clone(model)
                model_clone.set_params(**trial_params)

                scores = cross_val_score(
                    model_clone,
                    X_train,
                    y_train,
                    cv=3,
                    scoring="r2"
                )

                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=30, show_progress_bar=False)

            best_model = clone(model)
            best_model.set_params(**study.best_params)
            best_model.fit(X_train, y_train)

            test_score = r2_score(y_test, best_model.predict(X_test))

            report[model_name] = {
                "model": best_model,
                "score": test_score
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

'''
EVALUATING MODEL WITH GRIDSEARCHCV (BAD METHOD SO COMMENTED IT OUT)
'''
    
'''

# def evaluate_model(X_train, y_train, X_test, y_test, models, params):
#     try:
#         report = {}
#         for model_name, model in tqdm(models.items(), desc='Training Model'):
#             param_grid = params.get(model_name, {})

#             if not param_grid:
#                 model.fit(X_train, y_train)
#                 report[model_name] = {
#                     "model": model,
#                     "score": r2_score(y_test, model.predict(X_test))
#                 }
#                 continue
            
#             print(f"Running Optuna for {model_name}...")

#             def objective(trial):
#                 current_params = {}
#                 for param_name, param_options in param_grid.items():
#                     current_params[param_name] = trial.suggest_categorical(param_name, param_options)
                
#                 model.set_params(**current_params)

#                 cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='r2')

#                 return cv_scores.mean()
            
#             optuna.logging.set_verbosity(optuna.logging.WARNING)
#             study = optuna.create_study(direction="maximize")

#             best_params = study.best_params
#             print(f"Best params for {model_name}: {best_params}")
            
#             model.set_params(**best_params)
#             model.fit(X_train, y_train)

#             y_test_pred = model.predict(X_test)
#             score = r2_score(y_test, y_test_pred)

#             report[model_name] = {
#                 "model": model,
#                 "score": score
#             }

#             return report

#     except Exception as e:
#         raise CustomException(e, sys)

'''

'''
# def evaluate_model(X_train, y_train, X_test, y_test, models, params):
#     try:
#         report = {}

#         for model_name, model in tqdm(models.items(), desc="Training models", total=len(models)):
#             tqdm.write(f"Running GridSearch for {model_name}")
#             if model_name == "CatBoostRegressor":
#                 model.fit(X_train, y_train)
#                 score = r2_score(y_test, model.predict(X_test))
#                 report[model_name] = {
#                     "model": model,
#                     "score": score
#                 }
#                 continue

#             param_grid = params.get(model_name, {})

#             gs = GridSearchCV(
#                 model,
#                 param_grid,
#                 cv=3,
#                 verbose=0
#             )

#             gs.fit(X_train, y_train)

#             best_model = gs.best_estimator_

#             y_test_pred = best_model.predict(X_test)
#             score = r2_score(y_test, y_test_pred)

#             report[model_name] = {
#                 "model": best_model,
#                 "score": score
#             }

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
'''
