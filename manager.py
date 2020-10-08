import experiment as ex
from config import Config
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

if __name__ == "__main__":
    exp = ex.Experiment(label_rate = 0.8)
    for ds in Config.DATASET_NAME_LIST_TEST:
        try:
            exp.start(ds)
            exp.export_meta_features()
            exp.export_results()
        except:
            print("Failed for dataset: {}".format(ds))