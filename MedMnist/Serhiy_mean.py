import argparse
from pprint import pprint
import random
from copy import deepcopy
import csv
import datetime
import os
import pickle
from openpyxl import Workbook

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
from torch import nn
import torch.nn.functional as F

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.heuristics import Entropy
from baal.active.dataset import ActiveLearningDataset

import aug_lib
import numpy as np
from numpy import load
from baal_extended.ExtendedActiveLearningDataset import ExtendedActiveLearningDataset

import medmnist
from medmnist import INFO, Evaluator

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

import pandas as pd
import numpy as np
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD

import aug_lib
pjoin = os.path.join

"""
Minimal example to use BaaL.
# pip install baal
# conda activate deepAugmentEnv
# cd experiments
# python vgg_mcdropout_cifar10_org+aug_3_hochste-standartabweichung.py
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=50, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=100, type=int) # 1000, we will start training with only 1000(org)+1000(aug)=2000 labeled data samples out of the 50k (org) and
    parser.add_argument("--query_size", default=100, type=int)    # request 100(org)+100(aug)=200 new samples to be labeled at every cycle
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)     # 20 sampling for MC-Dropout to kick paths with low weights for optimization
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=5, type=int) # 20
    parser.add_argument("--augment", default=2, type=int)
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

# save uncertainties in pickle file
def generate_pickle_file(dt_string, active_set, epoch, oracle_indices, uncertainty):    
    pickle_filename = dt_string + (
        f"_uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.pkl"
    )
    dir_path = os.path.join(os.getcwd(), "uncertainties")
    isExist = os.path.exists("uncertainties")
    if not isExist:
        os.makedirs(dir_path)
    pickle_file_path = os.path.join(dir_path, pickle_filename)
    print("Saving file " + pickle_file_path)
    pickle.dump(
        {
            "oracle_indices": oracle_indices,
            "uncertainty": uncertainty,
            "labelled_map": active_set.labelled_map,
        },
        open(pickle_file_path, "wb")
    )
    return dir_path, pickle_file_path

def generate_excel_file(augment, dt_string, active_set, epoch, pickle_dir_path, df_lab_img): 
    excel_filename = dt_string + (
        f"_uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.xlsx"
    )
    excel_path = os.path.join(pickle_dir_path, excel_filename)

    uncertainties_std = df_lab_img.transpose()
    if augment == 1:
        uncertainties_std.columns = ['original', 'aug1', 'std']
    if augment == 2:   
        uncertainties_std.columns = ['original', 'aug1', 'aug2', 'std']

    uncertainties_std.to_excel(excel_path)


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

    # change dropout layer to MCDropout
    model = patch_module(model)

    if use_cuda:
        model.cuda()
    else: 
        print("WARNING! NO CUDA IN USE!")
    optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

    # Wraps the model into a usable API.
    model = ModelWrapper(model, criterion, replicate_in_memory=False)
    model.add_metric(name='accuracy', initializer=lambda : Accuracy())

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

    writer = SummaryWriter("vgg_mcdropout_cifar10_org+aug_3")    # baal-serhiy/experiments/vgg_mcdropout_cifar10_org+aug_3
    writer.add_custom_scalars(layout)

    for epoch in tqdm(range(args.epoch)):
        # if we are in the last round we want to train for longer epochs to get a more comparable result
        if epoch == (args.epoch - 1):
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
          with open(csv_filename, "a+", newline="") as out_file:
            csvwriter = csv.writer(out_file)
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
            probs = model.predict_on_dataset(
                pool,
                batch_size=hyperparams["batch_size"],
                iterations=hyperparams["iterations"],
                use_cuda=use_cuda,
            )

            if probs is not None and len(probs) > 0:
                # 1. Get uncertainty
                uncertainty = active_loop.heuristic.get_uncertainties(probs)
                oracle_indices = np.argsort(uncertainty)
                
                # Save pickle file for every tenth epoch
                if (epoch+1) % 10 == 0:
                    pickle_dir_path, pickle_file_path = generate_pickle_file(dt_string, active_set, epoch, oracle_indices, uncertainty)
                    mypickle = pd.read_pickle(pickle_file_path)

                if (hyperparams["augment"] != 1) and (hyperparams["augment"] != 2):
                    print("WARNING! Supporting only augmentation 1 and 2, for more write more code!")
                    sys.exit()
                if hyperparams["augment"] == 1:
                    orig_s2 = int((len(pool)/2)-1)
                    aug1_s1 = int(len(pool)/2)
                    aug1_s2 = int((len(pool)/2)*2-1)

                    original = uncertainty[0:orig_s2]
                    aug1 = uncertainty[aug1_s1:aug1_s2]

                    if len(original) != len(aug1):
                        # at least one list has a different length (take shorter and fill with 0 to match arrays equel length)
                        if len(original) > len(aug1):
                            aug1 += (len(original)-len(aug1)) * [0]
                        else:
                            original += (len(aug1)-len(original)) * [0]

                    matrix = np.vstack([original, aug1])
                if hyperparams["augment"] == 2: 
                    orig_s2 = int((len(pool)/3)-1)
                    aug1_s1 = int(len(pool)/3)
                    aug1_s2 = int((len(pool)/3)*2-1)
                    aug2_s1 = int((len(pool)/3)*2)
                    aug2_s2 = int(len(pool)-1)

                    original = uncertainty[0:orig_s2]
                    aug1 = uncertainty[aug1_s1:aug1_s2]
                    aug2 = uncertainty[aug2_s1:aug2_s2] 
                    print("3 original length "+str(len(original)))
                    print("4 aug1 length "+str(len(aug1)))
                    print("5 aug2 length "+str(len(aug2)))

                    if len(original) != len(aug1) or len(original) != len(aug2) or len(aug1) != len(aug2):
                        # at least one list has a different length (take shorter and fill with 0 to match arrays equel length)
                        if len(original) > len(aug1):
                            aug1 += (len(original)-len(aug1)) * [0]
                        else:
                            original += (len(aug1)-len(original)) * [0]
                        if len(original) > len(aug2):
                            aug2 += (len(original)-len(aug2)) * [0]
                        else:
                            original += (len(aug2)-len(original)) * [0]
                        if len(aug1) > len(aug2):
                            aug2 += (len(aug1)-len(aug2)) * [0]
                        else:
                            aug1 += (len(aug2)-len(aug1)) * [0]
                        
                    matrix = np.vstack([original, aug1, aug2])

                # 2. Calc standard deviation
                df_lab_img = pd.DataFrame(matrix)
                std_array = df_lab_img.mean()
                df_lab_img = pd.DataFrame(np.vstack([matrix, std_array]))
                
                # Save excel file for every tenth epoch
                if (epoch+1) % 10 == 0:
                    generate_excel_file(hyperparams["augment"], dt_string, active_set, epoch, pickle_dir_path, df_lab_img)
                
                # 3. Map std uncertainties to uncertainty array
                if hyperparams["augment"] == 1:
                    for i in range(len(uncertainty)):
                        uncertainty[i] = std_array[i % (len(pool)/2-1)]
                if hyperparams["augment"] == 2:
                    for i in range(len(uncertainty)):
                        uncertainty[i] = std_array[i % (len(pool)/3-1)]
                oracle_indices = np.argsort(uncertainty) # aufsteigend
                to_label = oracle_indices[::-1] # absteigend
                if len(to_label) > 0:
                    active_set.label(to_label[: hyperparams.get("query_size", 1)])
                else: break
            else:
                break
        else: 
            break

        train_accuracy = metrics["train_accuracy"].value
        test_accuracy = metrics["test_accuracy"].value
        train_loss = metrics["train_loss"].value
        test_loss = metrics["test_loss"].value

        logs = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_loss": train_loss,
            "test_loss": test_loss,
            #"labeled_data": active_set.labelled,
            "n_augmented_images_labelled": active_set.n_augmented_images_labelled,
            "n_unaugmented_images_labelled": active_set.n_unaugmented_images_labelled,
            "oracle_indices": oracle_indices,           # array([ 54130, 109117,   8053, ..., 146481,  32710,  16780])
            "uncertainty": uncertainty,                 # array([0.1608387 , 0.00041478, 0.00027927, ..., 0.01162097, 0.04465848, 0.00282653], dtype=float32)}
            "labelled_map": active_set.labelled_map,    # array([0, 0, 0, ..., 0, 0, 0]),
            "Next Training set size": len(active_set),
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
                train_loss,
                active_set.n_labelled,
                active_set.n_unaugmented_images_labelled,
                active_set.n_augmented_images_labelled
                )
            )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("accuracy/train", train_accuracy, epoch)
        writer.add_scalar("accuracy/test", test_accuracy, epoch)
        #writer.add_scalar("uncertainty", uncertainty, epoch)
    writer.close()


if __name__ == "__main__":
    main()