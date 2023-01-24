import pickle
import os
import sys #sys.exit()
import argparse
from pprint import pprint
import random
from copy import deepcopy
import csv
import datetime
import types

import argparse
from pprint import pprint
import random
from copy import deepcopy

import torch
import torch.nn as nn
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torchvision.models import vgg16
import torch.utils.data as data
from torchvision.transforms import transforms
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
from baal.utils.metrics import Accuracy

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.heuristics import Entropy
from baal.active.dataset import ActiveLearningDataset

from numpy import load
import aug_lib
import numpy as np
from baal_extended.ExtendedActiveLearningDataset import ExtendedActiveLearningDataset

import medmnist
from medmnist import INFO, Evaluator

"""
Minimal example to use BaaL.
"""


class ModifiedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __getitem__(self, index):
        x, y = self.original_dataset[index]
        y = y.item()  # convert the label tensor to a single integer
        return x, y

    def __len__(self):
        return len(self.original_dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=100, type=int)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=5, type=int)
    return parser.parse_args()


def get_datasets(initial_pool, as_rgb, download, python_class):
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    aug_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )

    DataClass = getattr(medmnist, python_class)

    train_ds = datasets.ImageFolder(
        "/home/erwe517e/08_mnist_aug_less_images/datasets/pathmnist/train", transform=transform
    )

    aug_train_ds = datasets.ImageFolder(
        "/home/erwe517e/08_mnist_aug_less_images/datasets/pathmnist/train", transform=aug_transform
    )

    test_set = datasets.ImageFolder(
        "/home/erwe517e/08_mnist_aug_less_images/datasets/pathmnist/test", transform=test_transform
    )

    # train_ds = DataClass(
    #    split="train", transform=transform, download=download, as_rgb=as_rgb
    # )

    # test_set = DataClass(
    #    split="test", transform=test_transform, download=download, as_rgb=as_rgb
    # )

    #data = load('/home/fue22/.medmnist/pathmnist.npz')
    #np_train_img = data["train_images"]
    #np_train_labels = data["train_labels"]
    #train_img = torch.from_numpy(np_train_img)
    #train_labels = torch.from_numpy(np_train_labels.flatten())

    ## Create a TensorDataset from the data and labels
    #train_ds = torch.utils.data.TensorDataset(train_img, train_labels)

    #transformed_train_dataset = torch.utils.data.MapDataset(train_ds, transform)

    #np_test_img = data["test_images"]
    #np_test_labels = data["test_labels"]
    #test_img = torch.from_numpy(np_test_img)
    #test_labels = torch.from_numpy(np_test_labels.flatten())

    ## Create a TensorDataset from the data and labels
    #test_ds = torch.utils.data.TensorDataset(test_img, test_labels)

    #transformed_test_dataset = torch.utils.data.MapDataset(test_ds, transform)
 

    # Note: We use the test set here as an example. You should make your own validation set.
    # train_ds = datasets.CIFAR10(
    #    ".", train=True, transform=transform, target_transform=None, download=True
    # )
    # test_set = datasets.CIFAR10(
    #    ".", train=False, transform=test_transform, target_transform=None, download=True
    # )

    #active_set = ActiveLearningDataset(train_ds)
    eald_set = ExtendedActiveLearningDataset(train_ds)
    eald_set.augment_n_times(2, augmented_dataset=aug_train_ds)

    # We start labeling randomly.
    eald_set.label_randomly(initial_pool)
    return eald_set, test_set


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    hyperparams = vars(args)
    data_flag = "pathmnist"
    info = INFO[data_flag]
    task = info["task"]
    as_rgb = True
    n_channels = 3 if as_rgb else info["n_channels"]
    n_classes = len(info["label"])
    download = True
    active_set, test_set = get_datasets(
        hyperparams["initial_pool"], as_rgb, download, info["python_class"]
    )

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hx%M")
    csv_filename = "uncertainties/metrics_cifarnet_" + dt_string + "_.csv"
    with open(csv_filename, "w+", newline="") as out_file:
      csvwriter = csv.writer(out_file)
      csvwriter.writerow(
        (
          "epoch",
          "test_acc",
          "train_acc",
          "test_loss",
          "train_loss",
          "Next training size",
          "amount original images labelled",
          "amount augmented images labelled"
        )
      )

    heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
    criterion = CrossEntropyLoss()
    model = vgg16(weights="VGG16_Weights.DEFAULT")
    model.classifier[6] = nn.Linear(4096, n_classes)
    # weights = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth")
    # weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
    # model.load_state_dict(weights, strict=False)

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()
    else: print("WARNING! NO CUDA IN USE!")

    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion)
    model.add_metric(name='accuracy',initializer=lambda : Accuracy())

    logs = {}
    logs["epoch"] = 0

    # for prediction we use a smaller batchsize
    # since it is slower
    active_loop = ActiveLearningLoop(
        active_set,
        model.predict_on_dataset,
        heuristic,
        hyperparams.get("query_size", 1),
        batch_size=10,
        iterations=hyperparams["iterations"],
        use_cuda=use_cuda,
    )
    # We will reset the weights at each active learning step.
    init_weights = deepcopy(model.state_dict())

    layout = {
        "Loss/Accuracy": {
            "Loss": ["Multiline", ["loss/train", "loss/test"]],
            "Accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
        },
    }

    writer = SummaryWriter("tb-results/testrun")
    writer.add_custom_scalars(layout)

    for epoch in tqdm(range(args.epoch)):
        # Load the initial weights.
        if epoch == (args.epoch - 1):
            hyperparams["learning_epoch"] = 75
        model.load_state_dict(init_weights)
        model.train_on_dataset(
            active_set,
            optimizer,
            hyperparams["batch_size"],
            hyperparams["learning_epoch"],
            use_cuda,
        )

        # Validation!
        model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
        metrics = model.metrics
        should_continue = active_loop.step()
        if not should_continue:
            break

        #pprint(model.get_metrics())
        test_loss = metrics["test_loss"].value
        train_loss = metrics["train_loss"].value
        test_accuracy = metrics["test_accuracy"].value
        train_accuracy = metrics["train_accuracy"].value
        logs = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "amount_labeled_data/next Training set size": active_set.n_labelled
        }
        pprint(logs)

        with open(csv_filename, "a+", newline="") as out_file:
            csvwriter = csv.writer(out_file)
            csvwriter.writerow(
                (
                epoch,
                test_accuracy,
                train_accuracy,
                test_loss,
                train_loss
                )
            )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("accuracy/test", test_accuracy, epoch)
    writer.close()
        
if __name__ == "__main__":
    main()
