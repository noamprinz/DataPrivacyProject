import os
import shutil

import matplotlib.pyplot as plt

from datasets import load_dataset

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision import transforms as transforms


BATCH_SIZE = 32
NUM_PARTITIONS = 10

def load_datasets(dataset_path, partition_id):
    # create dataset
    dataset = load_dataset("imagefolder", data_dir=dataset_path, drop_labels=False)
    # load partitioneer
    partitioner = IidPartitioner(num_partitions=NUM_PARTITIONS)
    # set dataset to partitioneer
    partitioner.dataset = dataset['train']
    # partition the dataset
    partition = partitioner.load_partition(partition_id=partition_id)
    # Divide Data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    return trainloader, valloader



def flwr_load_datasets(partition_id: int):
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": NUM_CLIENTS})
    partition = fds.load_partition(partition_id)
    # Divide Data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader


if __name__ == '__main__':
    dataset_path = "Data/bccd_dataset"

    trainloader, _ = load_datasets(dataset_path, partition_id=0)

    batch = next(iter(trainloader))
    images, labels = batch["image"], batch["label"]

    # Reshape and convert images to a NumPy array
    # matplotlib requires images with the shape (height, width, 3)
    images = images.permute(0, 2, 3, 1).numpy()

    # Denormalize
    images = images / 2 + 0.5

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(4, 8, figsize=(12, 6))

    # Loop over the images and plot them
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
        ax.axis("off")

    # Show the plot
    fig.tight_layout()
    plt.show()


    # trainloader, _, _ = load_datasets(partition_id=0)
    # batch = next(iter(trainloader))
    # images, labels = batch["img"], batch["label"]
#
    # # Reshape and convert images to a NumPy array
    # # matplotlib requires images with the shape (height, width, 3)
    # images = images.permute(0, 2, 3, 1).numpy()
#
    # # Denormalize
    # images = images / 2 + 0.5
#
    # # Create a figure and a grid of subplots
    # fig, axs = plt.subplots(4, 8, figsize=(12, 6))
#
    # # Loop over the images and plot them
    # for i, ax in enumerate(axs.flat):
    #     ax.imshow(images[i])
    #     ax.set_title(trainloader.dataset.features["label"].int2str([labels[i]])[0])
    #     ax.axis("off")
#
    # # Show the plot
    # fig.tight_layout()
    # plt.show()