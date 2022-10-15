import numpy as np
from sklearn.model_selection import train_test_split

import joblib
from datetime import datetime

import pipeline as pipe
from custom_preproc_classes.load_data import data_loading
from custom_preproc_classes.config.core import config


def training() -> None:
    """Train the model."""

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

    # fit model
    pipe.target_Val.fit(X_train, y_train)

    # persist trained model
    destination_file = config["pipeline_save_file_path"] + "_" + str(datetime.now())[:-7] + ".joblib"
    joblib.dump(pipe.target_Val, destination_file)


if __name__ == "__main__":
    training()
