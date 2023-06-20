import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import joblib
from datetime import datetime
import argparse
import os

import pipeline as pipe
from custom_preproc_classes.load_data import data_loading
from custom_preproc_classes.config.core import config
import logs
import mlflow



def parse_bool(bool_input: str):
    """
    This function allows to parse a boolian value when running in terminal.

    Args:
        - bool_input: Allowed values are "T", for True, or "F", for False
    """
    if bool_input == "T":
        return True
    elif bool_input == "F":
        return False
    else:
        raise ValueError("bool_input must be T (for True) or F (for False)")


def training(validation: str) -> None:
    """
    This function trains the final model by utilizing the pipeline.py module.

    Args:
        validation: If true model validation charateristics like R2 or mse will be calculated and a validation plot will be shown.
    """

    # ======= READ TRAINING DATA =======
    log = logs.setup_logger(file_name=config["logs_file_train"], logger_name=os.path.basename(__file__))
    try:
        data = data_loading(path_features=config["path_to_feature_file"], path_target=config["path_to_target_file"])
        log.info("Data loading successful.")
    except Exception as e:
        log.error("Error loading data", exc_info=e)

    # ==================================


    # ========= MODEL TRAINING =========
    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop([config["target"]], axis=1),  # predictors
        data[config["target"]],
        test_size=config["test_size"],
        # we are setting the random seed here
        # for reproducibility
        random_state=config["random_state"],
    )
    y_train = np.log(y_train)
    y_test = np.log(y_test)

    # ====== STORING RESULTS DATA =======
    try:
        # persist train and test data
        X_train.to_csv(config["folder_train_test_data"] + "X_train.csv", sep=";", header=True)
        X_test.to_csv(config["folder_train_test_data"] + "X_test.csv", sep=";", header=True)
        y_train.to_csv(config["folder_train_test_data"] + "y_train.csv", sep=";", header=True)
        y_test.to_csv(config["folder_train_test_data"] + "y_test.csv", sep=";", header=True)

        log.info("Training and test data successfully saved.")
    except Exception as e:
        log.error("Data could not be persisted due to error:", exc_info=e)

    # ---- MLFlow Run ---- #
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Best Practice")

    # Designated MLFlow run name
    run_name = "mlflow_run_" + "best_practice" + "_" + str(datetime.now())[:-7]
    print(run_name)

    # fit model
    with mlflow.start_run(run_name=run_name):
        try:
            log.info("Fitting pipeline to training data...")
            target_pipeline = pipe.target_Val.fit(X_train, y_train)
            log.info("Pipeline fitted successfully.")
        except Exception as e:
            log.error("Pipeline could not be fitted due to error:", exc_info=e)

        try:
            log.info("Prediction and mlflow versioning started...")
            pred_train = target_pipeline.predict(X_train)
            pred_test = target_pipeline.predict(X_test)

            performance_dict = {"mse_train": round(int(mean_squared_error(np.exp(y_train), np.exp(pred_train))), 2),
                                "mse_test": round(int(mean_squared_error(np.exp(y_test), np.exp(pred_test))), 2),
                                "rmse_train": round(int(mean_squared_error(np.exp(y_train), np.exp(pred_train), squared=False)), 2),
                                "rmse_test": round(int(mean_squared_error(np.exp(y_test), np.exp(pred_test), squared=False)), 2),
                                "r2_train": round(r2_score(np.exp(y_train), np.exp(pred_train)), 2),
                                "r2_test": round(r2_score(np.exp(y_test), np.exp(pred_test)), 2)
                                }

            # MLFlow loggings
            
            for key, value in performance_dict.items():
                mlflow.log_metric(key, value)

            for key, value in pipe.params.items(): 
                mlflow.log_param(key, value)

            #mlflow.log_artifact(train_data_path)
            #mlflow.log_artifact(test_data_path)
            mlflow.sklearn.log_model(sk_model=target_pipeline, artifact_path="model")
            log.info("Prediction and mlflow versioning performed successfully.")

        except Exception as e:
            log.error("Prediction and mlflow loggings could not be completed:", exc_info=e)

    

    # =====================================

    # ========= MODEL VALIDATION ==========
    # model validation if passed by the user
    validation = parse_bool(validation)

    if validation is True:
        # determine mse, rmse and r2 on the training data
        pred = pipe.target_Val.predict(X_train)

        print('train mse: {}'.format(int(
            mean_squared_error(np.exp(y_train), np.exp(pred)))))
        print('train rmse: {}'.format(int(
            mean_squared_error(np.exp(y_train), np.exp(pred), squared=False))))
        print('train r2: {}'.format(
            r2_score(np.exp(y_train), np.exp(pred))))
        print()

        # make predictions for test set
        pred = pipe.target_Val.predict(X_test)

        # determine mse, rmse and r2 on the test data
        print('test mse: {}'.format(int(
            mean_squared_error(np.exp(y_test), np.exp(pred)))))
        print('test rmse: {}'.format(int(
            mean_squared_error(np.exp(y_test), np.exp(pred), squared=False))))
        print('test r2: {}'.format(
            r2_score(np.exp(y_test), np.exp(pred))))
        print()

        # show real value vs. predicted as plot
        plt.scatter(np.exp(y_test), np.exp(pred))
        plt.xlabel('True target value')
        plt.ylabel('Predicted target value')
        plt.title('Evaluation of GBM Predictions')
        plt.show()
    # =======================================


if __name__ == "__main__":
    # add argument if model validation should be done
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validation", required=True, type=str, help="Use T for True and F for False")

    args = parser.parse_args()
    validation = args.validation

    training(validation=validation)
