from ..config.core import config
from .. import data_loading


def NAs_in_groups():
    data = data_loading(path_features=config["path_to_feature_file"], path_target=config["path_to_target_file"])

    assert sum(data["groups"].isna()) == 0


if __name__ == "__main__":
    NAs_in_groups()
