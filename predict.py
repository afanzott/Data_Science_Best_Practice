
import numpy as np
import pandas as pd

from custom_preproc_classes.preproc_predict import preproc_predict
from custom_preproc_classes.config.core import config

import joblib
import argparse


def make_prediction(input_data: pd.DataFrame) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = preproc_predict(input_data)
    target_pipeline = joblib.load(config["predict_pipeline"])

    predictions = target_pipeline.predict(X=data)
    results = [np.exp(pred) for pred in predictions]

    return results


if __name__ == "__main__":
    # add arguments for data file serving via terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", required=True)

    args = parser.parse_args()
    data_file = args.data_file

    print(make_prediction(input_data=data_file))
