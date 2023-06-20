import pandas as pd
import random
import argparse
from custom_preproc_classes.config.core import config



def create_prediction_sample(input_file: str):
    """
    This function create a random sample to test the prediction function

    Args:
        - input_file: Path to the respective data file to select a random sample from
    """
    # Read-in data file
    input_df = pd.read_csv(data_file, sep=";", index_col=False)

    # Select a random row index
    random_index = random.choice(input_df.index)

    # Get the row as a Series
    random_row = input_df.loc[random_index]

    # Create a new DataFrame with the random row and column headers
    prediction_sample = pd.DataFrame(random_row).transpose()

    # Store prediction sample to folder
    prediction_sample.to_csv(config["folder_prediction_samples"] + "prediction_sample_" + str(random_index) + ".csv", sep=";", header=True, index=False)


if __name__ == "__main__":
    # add arguments for data file serving via terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_file", required=True)

    args = parser.parse_args()
    data_file = args.data_file

    print(create_prediction_sample(input_file=data_file))
