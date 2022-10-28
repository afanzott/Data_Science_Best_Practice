import numpy as np
import pandas as pd

from custom_preproc_classes.load_data import data_loading_pred
from custom_preproc_classes.config.core import config
import logs

import joblib
import argparse
import os


def make_prediction(input_data: pd.DataFrame) -> dict:
    """Make a prediction using a saved model pipeline."""

    # ======= READ PREDICTION DATA =======
    log = logs.setup_logger(file_name=config["logs_file_predict"], logger_name=os.path.basename(__file__))
    try:
        data = data_loading_pred(input_data)
        log.info("Data loading successful.")
    except Exception as e:
        log.error("Error loading data", exc_info=e)

    # ====================================

    # =========== LOAD PIPELINE ==========
    try:
        target_pipeline = joblib.load(config["predict_pipeline"])
        log.info("Pipeline loaded successfully.")
    except Exception as e:
        log.error("Error loading pipeline", exc_info=e)
    # ====================================

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
