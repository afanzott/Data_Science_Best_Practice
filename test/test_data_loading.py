from custom_preproc_classes.config.core import config
from custom_preproc_classes.load_data import data_loading
import unittest


class Test_data_loading(unittest.TestCase):
    data = data_loading(path_features=config["path_to_feature_file"], path_target=config["path_to_target_file"])

    def test_nas_in_groups(self):
        self.assertEqual(sum(self.data["groups"].isna()), 0)

    def test_cat_vars_as_object(self):
        for col in self.data[config["cat_vars"]].columns:
            self.assertEqual(self.data[col].dtype, "object")

    def test_etherium_as_numeric(self):
        self.assertEqual(self.data[config["feat_to_numeric"]].dtype, "float64")


if __name__ == '__main__':
    unittest.main()
