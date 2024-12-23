import os
import shutil

import matplotlib.pyplot as plt

from datasets import load_dataset

from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision import transforms as transforms


BATCH_SIZE = 32

def load_local_datasets(partition_id, dataset_path, num_partitions):
    # create dataset
    dataset = load_dataset("imagefolder", data_dir=dataset_path, drop_labels=False)
    # get class names
    class_names = dataset['train'].features['label'].names
    class_dict = {i: class_names[i] for i in range(len(class_names))}
    # load partitioneer
    partitioner = IidPartitioner(num_partitions=num_partitions)
    # set dataset to partitioneer
    partitioner.dataset = dataset['train']
    # partition the dataset
    partition = partitioner.load_partition(partition_id=partition_id)
    # Divide Data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=16)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    test_set = dataset['test'].with_transform(apply_transforms)
    testloader = DataLoader(test_set, shuffle=True, batch_size=BATCH_SIZE)
    return trainloader, valloader, testloader, class_dict