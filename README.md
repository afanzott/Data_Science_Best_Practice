# RHI-DS-Challenge
Repo for the Data Science Best Practice Review

# Setup
Clone the repo and create the conda environment from the `Best_Practice_env.yaml` file via `conda env create -f Best_Practice_env.yaml` and activate it.

# Tests
Tests can be executed via `python -m unittest -v` from the root folder.

# Logging
Log files for training and prediction can be found in the logs folder.

# Model training
To train a model just run train.py. Currently it will use the specified files under the data folder (can be changed in the `config.yaml` file). To train with different data just swap the variables `path_to_feature_file` and `path_to_target_file` in the `config.yaml` to your data location.

The train.py file utilizes a preprocessing pipeline created in pipeline.py to prepare the data for training. Successfully trained models will be stored as .joblib pipelines under trained_models. You can change the variable `pipeline_save_file_path` in the `config_yaml` to your desired destination.

Additionally via the parameter -v you have to provide a boolean value to indicate if you want to validate your model and see respective plots (True) or not (False).

# Prediction
For making predictions you have to execute the predict.py file. But first change the variable `predict_pipeline` in the `config.yaml` to the file with your trained pipeline. Additionally you have to serve a data file as an argument when executing. Just add `-d path` where path corresponds to the location of the file, which should be predicted.
