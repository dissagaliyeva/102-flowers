import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import os

sns.set()


def batch(iterable, dictionary, n_batch, actual_order):
    """

    :param iterable:
    :param dictionary:
    :param n_batch:
    :param actual_order:
    :return:
    """
    # create a transformation to de-normalize the images
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])
    images, targets = next(iterable)
    images = invTrans(images)
    images = images.numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(30, 6))
    fig.suptitle(f'Batch of {n_batch}', fontsize=25)
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        img = np.transpose(np.clip(images[idx], 0, 1), (1, 2, 0))
        plt.imshow(img)
        label = dictionary[int(actual_order[targets[idx]])]
        ax.set_title(label, fontsize=14)


def hist(path='data/train'):
    df = get_folders(path)
    plt.figure(figsize=(25, 10))
    sns.displot(x='name', y='length', data=df)
    plt.tick_params(
        axis='x',
        which='both',
        bottom=False,
        labelbottom=False
    )
    plt.xlabel('folders')
    plt.show()


def get_folders(path):
    folders_dict = {}
    for folder in os.listdir(path):
        folders_dict[folder] = len(os.listdir(path + '/' + folder))
    return pd.DataFrame({'name': folders_dict.keys(), 'length': folders_dict.values()})
