import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import joblib
from datetime import datetime
import argparse

import pipeline as pipe
from custom_preproc_classes.load_data import data_loading
from custom_preproc_classes.config.core import config


def training(validation: bool) -> None:
    """
    This function trains the final model by utilizing the pipeline.py module.

    Args:
        validation: If true model validation charateristics like R2 or mse will be calculated and a validation plot will be shown.
    """

    # read training data
    data = data_loading(path_features=config["path_to_feature_file"], path_target=config["path_to_target_file"])

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

    # fit model
    pipe.target_Val.fit(X_train, y_train)

    # persist trained model
    destination_file = config["pipeline_save_file_path"] + "_" + str(datetime.now())[:-7] + ".joblib"
    joblib.dump(pipe.target_Val, destination_file)

    # persist train and test data
    X_train.to_csv(config["folder_train_test_data"] + "X_train_" + str(datetime.now())[:-7] + ".csv", sep=";", header=True)
    X_test.to_csv(config["folder_train_test_data"] + "X_test_" + str(datetime.now())[:-7] + ".csv", sep=";", header=True)

    y_train.to_csv(config["folder_train_test_data"] + "y_train_" + str(datetime.now())[:-7] + ".csv", sep=";", header=True)
    y_test.to_csv(config["folder_train_test_data"] + "y_test_" + str(datetime.now())[:-7] + ".csv", sep=";", header=True)

    # model validation if passed by the user
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


if __name__ == "__main__":
    # add argument if model validation should be done
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--validation", required=True, type=bool)

    args = parser.parse_args()
    validation = args.validation

    training(validation=validation)
