import experiment as ex

if __name__ == "__main__":
    exp = ex.Experiment(label_rate = 0.2)
    exp.start()
    exp.export_meta_features()
    exp.export_results()