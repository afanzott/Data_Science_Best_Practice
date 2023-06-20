import warnings
import argparse

import mlflow
from mlflow.tracking.client import MlflowClient
warnings.filterwarnings("ignore", category=FutureWarning)

from custom_preproc_classes.config.core import config


def parse_bool(to_production: str):
    if to_production == "T":
        return True
    elif to_production == "F":
        return False
    else:
        raise ValueError("to_production must be T or F")


def register_mlflow_model(run_id: str, to_production: str, experiment_name: str):
    """
    This function registers a model from an experiment specific run and sets the stage of the model to "Production".

    Args:
        - run_id: Specific id of the model's run
        - to_production: If True, the stage of the newly registered model is set to 'Production'.

    """

    # ---- Register model ---- #
    model_uri = "runs:/" + run_id + "/model"
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    model_details = mlflow.register_model(model_uri=model_uri, name=experiment_name)
    # print(*model_details, sep = "\n")

    to_production = parse_bool(to_production)

    # ---- Change stage of registered model to production ---- #
    if to_production:
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_details.name,
            version=model_details.version,
            stage="Production"
        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--runid", required=True, type=str)
    parser.add_argument("-to_p", "--to_production", required=True, type=str)
    parser.add_argument("-exn", "--experiment_name", required=True, type=str)
    args = parser.parse_args()
    run_id = args.runid
    to_production = args.to_production
    experiment_name = args.experiment_name

    print(run_id)
    print(to_production)

    register_mlflow_model(run_id=run_id, to_production=to_production, experiment_name=experiment_name)
