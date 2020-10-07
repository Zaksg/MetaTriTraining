import experiment as ex
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == "__main__":
    exp = ex.Experiment(label_rate = 0.8)
    exp.start()
    exp.export_meta_features()
    exp.export_results()