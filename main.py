import os
import json
import torch
import flwr

from typing import Dict, Optional, Tuple
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr.common import NDArrays, Scalar, Context
from flwr.client.mod import LocalDpMod

import model

from model import NewNet as Net
from data_util import load_local_datasets

DATASET_PATH = "Data/bccd_dataset"

# default simulation parameters
DEF_NUM_PARTITIONS = 3
DEF_NUM_ROUNDS = 5
DEF_NUM_EPOCHS = 1

# default dp parameters for simulation
DEF_DP_CLIPPING_NORM = 1.0
DEF_DP_SENSITIVITY = 1.0
DEF_DP_DELTA = 1e-5
DEF_EPSILON = 1.0

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU

print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")

BATCH_SIZE = 32


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return model.get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        model.set_parameters(self.net, parameters)
        # Train the network FOR NUM_EPOCHS number of epochs
        model.train(self.net, self.trainloader, epochs=NUM_EPOCHS)
        return model.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        model.set_parameters(self.net, parameters)
        loss, accuracy = model.test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    net = Net().to(DEVICE)
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, _, _ = load_local_datasets(partition_id, DATASET_PATH, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


def evaluate_server(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader, _ = load_local_datasets(0, DATASET_PATH, NUM_PARTITIONS)
    model.set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = model.test(net, testloader)
    if SAVE_MODEL:
        torch.save(net.state_dict(), f"{OUT_DIR}/{server_round}_model.pth")
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


class CustomStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = []
        self.server_evaluate_results = []

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result:
            self.server_evaluate_results.append({"round": server_round, "metrics": result[1]})
        return result

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)
        if aggregated:
            self.metrics.append({"round": server_round, "metrics": aggregated[1]})
        return aggregated

    def export_metrics(self):
        metrics_dict = {"server_evaluate_accuracy": self.server_evaluate_results,
                        "aggregated_evaluate_accuracy": self.metrics}
        return metrics_dict


def server_fn(context: Context) -> ServerAppComponents:
    strategy = CustomStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=NUM_PARTITIONS,
        evaluate_metrics_aggregation_fn=model.weighted_average,
        evaluate_fn=evaluate_server)
    # Configure the server for NUM_ROUNDS rounds of training
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


def run_single_simulation(out_dir, dp_mode=False, save_model=False, num_partitions=DEF_NUM_PARTITIONS,
                          num_rounds=DEF_NUM_ROUNDS, num_epochs=DEF_NUM_EPOCHS, epsilon=DEF_EPSILON,
                          clipping_norm=DEF_DP_CLIPPING_NORM):
    """
    Run a single simulation with the given parameters
    :param out_dir: output directory
    :param dp_mode: whether to run in differential privacy mode
    :param save_model: whether to save the model
    :param num_partitions: number of partitions
    :param num_rounds: number of rounds
    :param num_epochs: number of epochs
    :param epsilon: differential privacy epsilon
    :param clipping_norm: differential privacy clipping norm
    """
    # create the output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # initialize global params
    global OUT_DIR, SAVE_MODEL  # technical params
    OUT_DIR = out_dir
    SAVE_MODEL = save_model
    global NUM_ROUNDS, NUM_EPOCHS, NUM_PARTITIONS, EPSILON, CLIPPING_NORM  # simulation params
    NUM_ROUNDS = num_rounds
    NUM_EPOCHS = num_epochs
    NUM_PARTITIONS = num_partitions
    EPSILON = epsilon
    CLIPPING_NORM = clipping_norm
    # create param dict
    param_dict = {"simulation_name": os.path.basename(OUT_DIR), "simulation_path": OUT_DIR, "num_rounds": NUM_ROUNDS,
                  "num_epochs": NUM_EPOCHS, "num_partitions": NUM_PARTITIONS, "dp_mode": dp_mode,
                  "save_model": SAVE_MODEL}
    if dp_mode:
        param_dict["epsilon"] = EPSILON
        param_dict["clipping_norm"] = CLIPPING_NORM
    # save the param dict
    with open(f"{out_dir}/params.json", "w") as f:
        json.dump(param_dict, f)
    # Create the ClientApp
    if not dp_mode:
        client = ClientApp(client_fn=client_fn)
    else:
        # if differential privacy mode is enabled, create a LocalDpMod object with EPSILON value for epsilon
        local_dp_obj = LocalDpMod(clipping_norm=CLIPPING_NORM, sensitivity=DEF_DP_SENSITIVITY, epsilon=EPSILON,
                                  delta=DEF_DP_DELTA)
        client = ClientApp(client_fn=client_fn, mods=[local_dp_obj])
    # Create an instance of the model and get the parameters
    server = ServerApp(server_fn=server_fn)
    # Run the simulation
    print("Running simulation for the following parameters:")
    print(f"Number of partitions: {NUM_PARTITIONS}, Number of rounds: {NUM_ROUNDS}, Number of epochs: {NUM_EPOCHS}, "
          f"DP Epsilon: {EPSILON}, clipping norm: {CLIPPING_NORM}")
    print(f"Output directory: {OUT_DIR}, Saving model: {SAVE_MODEL}, Differential Privacy mode: {dp_mode}")
    # Run the simulation with NUM_PARTITIONS number of clients
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS)
    print(f"##### Simulation completed #####")
    output_metrics = server._strategy.export_metrics()
    # export metrics
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(output_metrics, f)


def analyze_num_epochs(out_dir, num_epochs_list):
    """
    Analyze the effect of number of epochs on the simulation
    """
    for epoch in num_epochs_list:
        epochs_out_dir = f"{out_dir}/num_epochs_{epoch}"
        run_single_simulation(epochs_out_dir, dp_mode=False, save_model=False, num_epochs=epoch)


def analyze_num_rounds(out_dir, num_rounds_list, num_epochs=DEF_NUM_EPOCHS):
    """
    Analyze the effect of number of rounds on the simulation
    """
    for round in num_rounds_list:
        rounds_out_dir = f"{out_dir}/num_rounds_{round}"
        run_single_simulation(rounds_out_dir, dp_mode=False, save_model=False, num_rounds=round,
                              num_epochs=num_epochs)


def analyze_num_partitions(out_dir, num_partitions_list, num_rounds=DEF_NUM_ROUNDS, num_epochs=DEF_NUM_EPOCHS):
    """
    Analyze the effect of number of partitions on the simulation
    """
    for partition in num_partitions_list:
        partition_out_dir = f"{out_dir}/num_partitions_{partition}"
        run_single_simulation(partition_out_dir, dp_mode=False, save_model=False, num_partitions=partition,
                              num_rounds=num_rounds, num_epochs=num_epochs)


def analyze_epsilons(out_dir, epsilons, num_rounds=DEF_NUM_ROUNDS, num_epochs=DEF_NUM_EPOCHS,
                     num_partitions=DEF_NUM_PARTITIONS,
                     clipping_norm=DEF_DP_CLIPPING_NORM):
    """
    Analyze the effect of epsilon on the simulation
    """
    for epsilon in epsilons:
        epsilon_out_dir = f"{out_dir}/epsilon_{epsilon}"
        run_single_simulation(epsilon_out_dir, dp_mode=True, save_model=False, num_partitions=num_partitions,
                              num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                              clipping_norm=clipping_norm)


def analyze_clipping_norms(out_dir, clipping_norms, num_rounds=DEF_NUM_ROUNDS, num_epochs=DEF_NUM_EPOCHS,
                           num_partitions=DEF_NUM_PARTITIONS, epsilon=DEF_EPSILON):
    """
    Analyze the effect of epsilon on the simulation
    """
    for clipping_norm in clipping_norms:
        clipping_norm_out_dir = f"{out_dir}/clipping_norm_{clipping_norm}"
        run_single_simulation(clipping_norm_out_dir, dp_mode=True, save_model=False, num_partitions=num_partitions,
                              num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                              clipping_norm=clipping_norm)


def run_quantitive_analysis_simulations(out_dir):
    print(f"##### Analyzing Number of Epochs #####")
    num_epochs_list = [1, 2, 3, 4, 5, 10]
    analyze_num_epochs(out_dir, num_epochs_list)
    # set the number of epochs to 5
    NUM_EPOCHS = 5
    print(f"##### Analyzing Number of Rounds #####")
    num_rounds_list = [1, 3, 5, 10, 15, 20]
    analyze_num_rounds(out_dir, num_rounds_list, num_epochs=NUM_EPOCHS)
    NUM_ROUNDS = 5
    print(f"##### Analyzing Number of Partitions #####")
    num_partitions_list = [1, 2, 3, 4, 5, 10, 20]
    analyze_num_partitions(out_dir, num_partitions_list, num_rounds=NUM_ROUNDS, num_epochs=NUM_EPOCHS)
    NUM_PARTITIONS = 3
    print(f"##### Analyzing Epsilon with Clipping Norm set to 1 #####")
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    CLIPPING_NORM = 1
    analyze_epsilons(out_dir, epsilons, num_rounds=NUM_ROUNDS, num_epochs=NUM_EPOCHS, num_partitions=NUM_PARTITIONS,
                     clipping_norm=CLIPPING_NORM)
    print(f"##### Analyzing Epsilon with Clipping Norm set to inf #####")
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    CLIPPING_NORM = float("inf")
    analyze_epsilons(out_dir, epsilons, num_rounds=NUM_ROUNDS, num_epochs=NUM_EPOCHS, num_partitions=NUM_PARTITIONS,
                     clipping_norm=CLIPPING_NORM)
    EPSILON = 1
    print("##### Analyzing Clipping Norm #####")
    clipping_norms = [float("inf"), 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    analyze_clipping_norms(out_dir, clipping_norms, num_rounds=NUM_ROUNDS, num_epochs=NUM_EPOCHS,
                           num_partitions=NUM_PARTITIONS, epsilon=EPSILON)


def run_qualitative_analysis_simulations(out_dir):
    # fixed values
    num_epochs = 5
    num_rounds = 5
    num_partitions = 3
    # run baseline experiment
    print(f"##### Running Baseline Experiment #####")
    baseline_out_dir = f"{out_dir}/baseline"
    clipping_norm = float("inf")
    epsilon = 1e5
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(baseline_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run weak epsilon experiment
    print(f"##### Running Weak Epsilon Experiment #####")
    weak_epsilon_out_dir = f"{out_dir}/weak_epsilon"
    clipping_norm = float("inf")
    epsilon = 10
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(weak_epsilon_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run strong epsilon experiment
    print(f"##### Running Strong Epsilon Experiment #####")
    strong_epsilon_out_dir = f"{out_dir}/strong_epsilon"
    clipping_norm = float("inf")
    epsilon = 0.1
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(strong_epsilon_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run weak clipping norm experiment
    print(f"##### Running Weak Clipping Norm Experiment #####")
    weak_clipping_norm_out_dir = f"{out_dir}/weak_clipping_norm"
    clipping_norm = 25
    epsilon = 1e5
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(weak_clipping_norm_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run strong clipping norm experiment
    print(f"##### Running Strong Clipping Norm Experiment #####")
    strong_clipping_norm_out_dir = f"{out_dir}/strong_clipping_norm"
    clipping_norm = 10
    epsilon = 1e5
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(strong_clipping_norm_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run weak combination experiment
    print(f"##### Running Weak Combination Experiment #####")
    weak_combination_out_dir = f"{out_dir}/weak_combination"
    clipping_norm = 25
    epsilon = 10
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(weak_combination_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)
    # run strong combination experiment
    print(f"##### Running Strong Combination Experiment #####")
    strong_combination_out_dir = f"{out_dir}/strong_combination"
    clipping_norm = 10
    epsilon = 0.1
    print(
        f"Running simulation with epsilon: {epsilon}, clipping norm: {clipping_norm}, num_partitions: {num_partitions}, "
        f"num_rounds: {num_rounds}, num_epochs: {num_epochs}")
    run_single_simulation(strong_combination_out_dir, dp_mode=True, save_model=True, num_partitions=num_partitions,
                          num_rounds=num_rounds, num_epochs=num_epochs, epsilon=epsilon,
                          clipping_norm=clipping_norm)


def main(out_dir):
    run_quantitive_analysis_simulations(out_dir)
    run_qualitative_analysis_simulations(out_dir)


if __name__ == '__main__':
    OUT_DIR = 'SimulationOutputs'
    main(OUT_DIR)
