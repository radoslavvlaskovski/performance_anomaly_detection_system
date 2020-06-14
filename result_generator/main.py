from performance_anomaly_detection.result_generator.experiment import ExperimentData, Experiment
from performance_anomaly_detection.result_generator.train import ExperimentModel, ExperimentModelType, \
    ExperimentModelFile
from sklearn.ensemble import RandomForestRegressor
from performance_anomaly_detection.result_generator.prepare import DataPrepare


def create_data():
    data_prepare = DataPrepare("../data_collector/out/pl-20-2/tracing_data.csv",
                               "../data_collector/out/pl-20-2/currencyservice-0.csv")
    data_prepare.prepare_tracing_data()
    data_prepare.prepare_metrics_data()
    data_prepare.combine_and_save("data/pl-20-2")


def run():
    experiment_data = ExperimentData("data/pl-20-2/metrics_data.csv", "data/pl-20-2/spans_data.csv", "data/pl-20-2/traces_data.csv")
    experiment = Experiment(experiment_data, "results/pl-20-2")
    print("Starting experiemnt: %s" % str(experiment.id))

    rf_model = ExperimentModel("rf", ExperimentModelType.rf,
                               RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=25,
                                                     min_samples_leaf=5, min_samples_split=2, n_estimators=50,
                                                     n_jobs=2, random_state=1, verbose=True))
    lstm_model = ExperimentModel("lstm", ExperimentModelType.lstm, {"layers_size": [12, 12]})
    filter_net_model = ExperimentModel("filter_net", ExperimentModelType.filternet,
                                       {"layer_one_size": 4, "layer_two_size": 4, "exog_layer_sizes": [8, 8]})
    experiment.add_model(filter_net_model)
    experiment.add_model(rf_model)
    experiment.add_model(lstm_model)
    experiment.run()


def run_old():
    experiment_data = ExperimentData("data/120-220/metrics_data.csv", "data/120-220/spans_data.csv",
                                     "data/120-220/traces_data.csv")
    experiment = Experiment(experiment_data, "results/120-220")
    print("Starting experiemnt: %s" % str(experiment.id))

    rf_model = ExperimentModelFile().load_model("rf_trained",
                                                ExperimentModelType.rf,
                                                "results/175/86b43168-f527-44bb-98aa-9ea4ccae2a26/rf/ExperimentDataType.metrics_model")

    filter_net = ExperimentModelFile().load_model("filternet_trained",
                                                  ExperimentModelType.filternet,
                                                  "results/175/86b43168-f527-44bb-98aa-9ea4ccae2a26/filter_net/ExperimentDataType.metrics_model.h5")
    lstm = ExperimentModelFile().load_model("lstm_trained",
                                            ExperimentModelType.lstm,
                                            "results/175/86b43168-f527-44bb-98aa-9ea4ccae2a26/lstm/ExperimentDataType.metrics_model.h5")
    experiment.add_model(filter_net)
    experiment.add_model(rf_model)
    experiment.add_model(lstm)
    experiment.run()


if __name__ == "__main__":
    run()
