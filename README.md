# Best-Practice-Data-Science
Repo for the Data Science Best Practice Review (Expert Review)

# Setup
Clone the repo and create the conda environment from the `Best_Practice_env.yaml` file via `conda env create -f Best_Practice_env.yaml` and activate it.

# Tests
Tests can be executed via `python -m unittest -v` from the root folder.

# Logging
Log files for training and prediction can be found in the logs folder.

# Model training
To train a model just run train via the respective make command. Currently it will use the specified files under the data folder (can be changed in the `config.yaml` file). To train with different data just swap the variables `path_to_feature_file` and `path_to_target_file` in the `config.yaml` to your data location.

The train.py file utilizes a preprocessing pipeline created in pipeline.py to prepare the data for training. Successfully trained models will be stored in the `mlruns.db` and `mlruns` folder. 

Additionally via the parameter -v you have to provide a T for True and F for False to indicate if you want to validate your model and see respective plots (True) or not (False).

# Prediction
MLFlow is used to first register a model to the production stage. The script `register_mlflow_model.py` uses a `run_id` of a given mlflow run and stages this run's model to a given stage. Only models that are on stage `production` will be used for the prediction.

For making predictions you have to execute the predict.py file via the respective make command. The script uses the information from mlflow to use a trained model in production to make a prediction. You only have to serve a data file as an argument when executing. Just add `-d path` where path corresponds to the location of the file, which should be predicted. You can use example inputs from the `prediction_samples` folder in the `data` directory to test your predictions. If you would like to have more sample inputs, execute the script `create_prediction_sample.py`. It randomly selects a row from the raw dataset and converts it to a respective input for the prediction. You only need to provide the script with the path to the raw dataset in the `data` folder.

# Viewing the results on MLFlow
To view the results of the trained models use `mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000 &` and go to http://127.0.0.1:5000/
To stop the running server use `CTRL+C` and type: `pkill -f unicorn`

# Make
The repository's `Makefile` contains certain rules to facilitate the execution of commands. You can add any additional rule to it if you want. Execute the rules simply by adding `make` before the rule (i.e. `make train`) Here are some examples:
- train: trains a new model and provides a validation of the results (change the argument to F if you do not wish to see the validation results)
- predict: predicts the outcome of the registered model for a new data input (change the path of the data input if necessary)
