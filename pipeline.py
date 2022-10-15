# from Scikit-learn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel

# from feature-engine
from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
)

from feature_engine.encoding import (
    RareLabelEncoder,
    OrdinalEncoder
)
from feature_engine.transformation import LogTransformer

from custom_preproc_classes import custom_preproc as pp
from custom_preproc_classes.config.core import config


params = {
    "n_estimators": config["n_estimators"],
    "max_depth": config["max_depth"],
    "min_samples_split": config["min_samples_split"],
    "learning_rate": config["learning_rate"],
    "loss": config["loss"],
    "verbose": config["verbose"],
    "random_state": config["random_state"]
}

# create pipeline
target_Val = Pipeline([

    # == TEMPORAL VARIABLES ====
    # impute NA for false dates
    ('temp_var_step1', pp.TempVarMissingTransformer(
        temp_vars=config["temporal_vars_step1"])),

    # calc elapsed time
    ('temp_var_step2_1', pp.TempVarElapsedTimeTransformer(
        name=config["temporal_vars_step2_1"][0], var1=config["temporal_vars_step2_1"][1], var2=config["temporal_vars_step2_1"][2])),
    ('temp_var_step2_2', pp.TempVarElapsedTimeTransformer(
        name=config["temporal_vars_step2_2"][0], var1=config["temporal_vars_step2_2"][1], var2=config["temporal_vars_step2_2"][2])),
    ('temp_var_step2_3', pp.TempVarElapsedTimeTransformer(
        name=config["temporal_vars_step2_3"][0], var1=config["temporal_vars_step2_3"][1], var2=config["temporal_vars_step2_3"][2])),

    # split vars to year, month, day, hour, minute, second
    ('temp_var_step3_date', pp.TempVarSplitTransformer(
        vars=config["temporal_vars_step3_date"], date_or_dt="date")),
    ('temp_var_step3_datetime', pp.TempVarSplitTransformer(
        vars=config["temporal_vars_step3_datetime"], date_or_dt="datetime")),

    # ===== IMPUTATION =====
    # add missing indicator
    ('missing_indicator', AddMissingIndicator(variables=config["numerical_vars_with_na"])),

    # impute numerical variables with the mean
    ('mean_imputation', MeanMedianImputer(
        imputation_method='mean', variables=config["numerical_vars_with_na"]
    )),

    # ==== LOG TRANSFORMATION =====
    ('log', LogTransformer(variables=config["numerical_log_vars"])),

    # == CATEGORICAL ENCODING
    ('rare_label_encoder', RareLabelEncoder(
        tol=0.01, n_categories=1, variables=config["categorical_vars_others"]
    )),

    # encode categorical and discrete variables using the target mean
    ('categorical_encoder', OrdinalEncoder(
        encoding_method='ordered', variables=config["categorical_vars_others"])),

    # ==== SCALING VARIABLES TRANSFORMATION =====
    ('scaler', MinMaxScaler()),

    # ==== VARIABLE SELECTION =====
    ('selector', SelectFromModel(Lasso(alpha=config["alpha"], random_state=config["random_state"]))),
    ('GBM', GradientBoostingRegressor(**params)),
])
