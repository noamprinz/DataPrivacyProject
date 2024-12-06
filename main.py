from typing import Dict, Optional, Tuple
import os
import sys
import json
import torch

import flwr

from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, DifferentialPrivacyClientSideFixedClipping
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
from flwr.client.mod import LocalDpMod

import model
from data_util import load_local_datasets

DATASET_PATH = "Data/bccd_dataset"

from model import NewNet as Net, set_parameters, test

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

    # TODO: replace global variable with context variable

    trainloader, valloader, _, _ = load_local_datasets(partition_id, DATASET_PATH, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


def evaluate_server(
        server_round: int,
        parameters: NDArrays,
        config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader, _ = load_local_datasets(0, DATASET_PATH, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
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
    # Configure the server for 3 rounds of training
    # TODO: CONTROL ROUNDS
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


def run_single_simulation(out_dir, dp_mode=False):
    """
    Run a single simulation with the given number of partitions
    :param:
    :return:
    """
    # create the output directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # initialize global params
    global OUT_DIR, NUM_ROUNDS, NUM_EPOCHS, NUM_PARTITIONS
    OUT_DIR = out_dir
    NUM_ROUNDS = 1
    NUM_EPOCHS = 1
    NUM_PARTITIONS = 1
    # create param dict
    param_dict = {"simulation_name":os.path.basename(out_dir), "simulation_path":out_dir, "num_rounds": NUM_ROUNDS,
                  "num_epochs": NUM_EPOCHS, "num_partitions": NUM_PARTITIONS, "dp_mode": dp_mode}
    # TODO: Add "save model" flag
    # save the param dict
    with open(f"{out_dir}/params.json", "w") as f:
        json.dump(param_dict, f)
    # Create the ClientApp
    if not dp_mode:
        client = ClientApp(client_fn=client_fn)
    else:
        # TODO: Set default values
        local_dp_obj = LocalDpMod(0.5, 0.5, 0.01, 0.5)
        client = ClientApp(client_fn=client_fn, mods=[local_dp_obj])
    # Create an instance of the model and get the parameters
    server = ServerApp(server_fn=server_fn)
    # Run the simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS)
    print(f"##### Simulation completed #####")
    output_metrics = server._strategy.export_metrics()
    # export metrics
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(output_metrics, f)


def main(out_dir):
    print(f"##### Running full simulation #####")
    simulation_out_dir = os.path.join(out_dir, "full_simulation")
    run_single_simulation(simulation_out_dir)


if __name__ == '__main__':
    OUT_DIR = 'SimulationOutputs/try_single_client'
    main(OUT_DIR)
