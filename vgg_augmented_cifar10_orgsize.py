import argparse
import csv
import datetime
import os
import pickle
import random

import argparse
from pprint import pprint
import random
import csv
from copy import deepcopy
from time import time
import datetime
import numpy as np

from tensorboardX import SummaryWriter
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.dataset import ActiveLearningDataset

import aug_lib

from baal_extended.ExtendedActiveLearningDataset import ExtendedActiveLearningDataset

pjoin = os.path.join


"""
Minimal example to use BaaL.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--query_size", default=50, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=5, type=int)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    aug_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )

    aug_train_ds = datasets.CIFAR10(
        ".", train=True, transform=aug_transform, target_transform=None, download=True
    )
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )
    eald_set = ExtendedActiveLearningDataset(train_ds)
    # active_set = ActiveLearningDataset(
    #    train_ds, pool_specifics={"transform": test_transform}
    # )
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

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hx%M")
    with open("results/csv/metrics_cifarnet_" + dt_string + "_.csv", "w+", newline="") as out_file:
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

        hyperparams = vars(args)

        active_set, test_set = get_datasets(hyperparams["initial_pool"])

        heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
        criterion = CrossEntropyLoss()
        model = vgg16(weights="VGG16_Weights.DEFAULT")
        #anpassen Model an zehn Klassen
        model.classifier[6] = nn.Linear(4096, 10)

        # change dropout layer to MCDropout
        model = patch_module(model)

        if use_cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

        # Wraps the model into a usable API.
        model = ModelWrapper(model, criterion, replicate_in_memory=False)
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
            # save uncertainties into one file per epoch
            uncertainty_folder="results/uncertainties",
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

        tensorboardwriter = SummaryWriter("results/tensorboard/tb-results" + dt_string + "/testrun")
        tensorboardwriter.add_custom_scalars(layout)

        for epoch in tqdm(range(args.epoch)):
            # if we are in the last round we want to train for longer epochs to get a more comparable result
            if epoch == args.epoch:
                hyperparams["learning_epoch"] = 75
            
            # Load the initial weights.
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
            
            # get origin amount of labelled augmented/unaugmented images
            if(epoch == 0):
                csvwriter.writerow(
                    (
                        -1,
                        0,
                        0,
                        0,
                        0,
                        active_set.n_labelled,
                        active_set.n_unaugmented_images_labelled,
                        active_set.n_augmented_images_labelled
                    )
                )

            # replacement for step
            pool = active_set.pool
            if len(pool) > 0:
                indices = np.arange(len(pool)) # array von 0 bis len(pool)-1 (nach initial label: 146999)
                probs = model.predict_on_dataset(pool,batch_size=10,iterations=hyperparams["iterations"],use_cuda=use_cuda)
                #if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                # -> "isinstance(...) needed when using predict_..._Generator"
                if probs is not None and len(probs) > 0:
                    to_label, uncertainty = heuristic.get_ranks(probs) 
                    # to_label -> indices sortiert von gr????ter zu niedrigster uncertainty
                    # uncertainty -> alle uncertainties des pools 
                    to_label = indices[np.array(to_label)] # was hier passiert keine Ahnung aber to_label bleibt gleich also unn??tze Zeile?
                    if len(to_label) > 0:
                        active_set.label(to_label[: hyperparams.get("query_size", 1)])
                    else: break
                else:
                    break
            else: 
                break
            
            # suggested solution from baal-dev but works with the whole dataset and I think we should use the pool and have to translate the indices afterwards (like above in replacement for step)
            #####
            predictions = model.predict_on_dataset(active_set._dataset,
                                                    hyperparams["batch_size"],
                                                    hyperparams["iterations"],
                                                    use_cuda) 
            uncertainty = BALD().get_uncertainties(predictions)
            oracle_indices = uncertainty.argsort()
            #####

            #should_continue = active_loop.step()
            #if not should_continue:
            #    break
            
            test_acc = metrics["test_accuracy"].value
            train_acc = metrics["train_accuracy"].value
            test_loss = metrics["test_loss"].value
            train_loss = metrics["train_loss"].value 

            if epoch % 10 == 0:
                predictions = model.predict_on_dataset(
                    active_set._dataset,
                    batch_size=hyperparams["batch_size"],
                    iterations=hyperparams["iterations"],
                    use_cuda=use_cuda,
                )
                uncertainty = active_loop.heuristic.get_uncertainties(predictions)
                # save uncertainty and label map to csv
                oracle_indices = np.argsort(uncertainty)
                active_set.labelled_map
                uncertainty_name = (
                    f"uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.pkl"
                )
                pickle.dump(
                    {
                        "oracle_indices": oracle_indices,
                        "uncertainty": uncertainty,
                        "labelled_map": active_set.labelled_map,
                    },
                    open(pjoin("uncertainties_augmented_cifarnet", uncertainty_name), "wb"),
                )


            logs = {
                "epoch": epoch,
                "test_acc": test_acc,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "train_loss": train_loss,
                "Next training size": active_set.n_labelled,
                "amount original images labelled": active_set.n_augmented_images_labelled,
                "amount augmented images labelled": active_set.n_unaugmented_images_labelled,
            }
            pprint(logs)

            csvwriter.writerow(
                (
                    epoch,
                    test_acc,
                    train_acc,
                    test_loss,
                    train_loss,
                    active_set.n_labelled,
                    active_set.n_unaugmented_images_labelled,
                    active_set.n_augmented_images_labelled
                )
            )

            tensorboardwriter.add_scalar("loss/train", train_loss, epoch)
            tensorboardwriter.add_scalar("loss/test", test_loss, epoch)
            tensorboardwriter.add_scalar("accuracy/train", train_acc, epoch)
            tensorboardwriter.add_scalar("accuracy/test",test_acc, epoch)
            #tensorboardwriter.add_scalar("accuracy/train", metrics["validation_accuracy"].value, epoch)
            #tensorboardwriter.add_scalar("accuracy/test", metrics["validation_accuracy"].value, epoch)
        tensorboardwriter.close()


if __name__ == "__main__":
    main()
