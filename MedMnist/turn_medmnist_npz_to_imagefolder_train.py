import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import load

# Convert the data and labels to PyTorch tensors

data = load("/home/erwe517e/.medmnist/pathmnist.npz")
np_img = data["train_images"]
#np_img=np_img[0:50000]
np_labels = data["train_labels"]
#np_labels=np_labels[0:50000]


# Create a directory for each unique label
unique_labels = np.unique(np_labels.flatten())
for label in unique_labels:
    os.makedirs("/home/erwe517e/08_mnist_aug_less_images/datasets/pathmnist/train/" + str(label), exist_ok=True)
i = 0
# Save each image to the appropriate directory
for image, label in zip(np_img, np_labels.flatten()):
    i += 1
    # np.save("/home/fue22/datasets/pathmnist/"+label+"/image_"+i+".npy", image)
    plt.imsave(
        "/home/erwe517e/08_mnist_aug_less_images/datasets/pathmnist/train/"
        + str(label)
        + "/image_"
        + str(i)
        + ".png",
        image,
    )
