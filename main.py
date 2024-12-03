from typing import Dict, Optional, Tuple
import os
import torch

import flwr

from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdagrad
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context

import model
from data_util import load_local_datasets

DATASET_PATH = "Data/bccd_dataset"

from model import Net, DEVICE, DATASET_PATH, NUM_PARTITIONS, set_parameters, test

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
        #TODO: CONTROL EPOCHS
        model.train(self.net, self.trainloader, epochs=1)
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

    #TODO: replace global variable with context variable

    trainloader, valloader, _ = load_local_datasets(partition_id, DATASET_PATH, num_partitions)
    return FlowerClient(partition_id, net, trainloader, valloader).to_client()


def evaluate_server(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    net = Net().to(DEVICE)
    _, _, testloader = load_local_datasets(0, DATASET_PATH, NUM_PARTITIONS)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    torch.save(net.state_dict(), f"{OUT_DIR}/{server_round}_model.pth")
    print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    return loss, {"accuracy": accuracy}


def server_fn(context: Context) -> ServerAppComponents:
    # Create the FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1,
        fraction_evaluate=0.5,
        min_fit_clients=5,
        min_evaluate_clients=3,
        min_available_clients=NUM_PARTITIONS,
        evaluate_metrics_aggregation_fn=model.weighted_average,
        evaluate_fn=evaluate_server)  # Pass the evaluation function
    # Configure the server for 3 rounds of training
    config = ServerConfig(num_rounds=3)
    return ServerAppComponents(strategy=strategy, config=config)


def main():
    for num_partitions in NUM_PARTITIONS_LIST:
        global NUM_PARTITIONS, OUT_DIR
        NUM_PARTITIONS = num_partitions
        OUT_DIR = f"Models/{NUM_PARTITIONS}_partitions"
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        print(f"##### Running simulation with {NUM_PARTITIONS} partitions #####")
        run_single_simulation()


def run_single_simulation():
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)
    # Create an instance of the model and get the parameters
    server = ServerApp(server_fn=server_fn)
    # Run the simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS)


if __name__ == '__main__':
    DEF_NUM_PARTITIONS = 5
    NUM_PARTITIONS_LIST = [5, 10]
    main()


