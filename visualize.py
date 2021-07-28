import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import os
import plotly.express as px

sns.set()


def batch(iterable, dictionary, n_batch, actual_order, cuda=False, model=None):
    """

    :param model:
    :param cuda:
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

    if cuda:
        images = images.cuda()

    if model is not None:
        output = model(images)
        _, pred = torch.max(output, 1)
        pred = np.squeeze(pred.numpy()) if not cuda else np.squeeze(pred.cpu().numpy())

    images = invTrans(images)
    images = images.numpy()

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(40, 10))
    fig.suptitle(f'Batch of {n_batch}', fontsize=25)
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        label = dictionary[int(actual_order[targets[idx]])]

        if model is not None:
            if cuda:
                images[idx] = images[idx].cpu()
            img = np.transpose(np.clip(images[idx], 0, 1), (1, 2, 0))
            plt.imshow(img)
            pred_label = dictionary[int(actual_order[pred[idx]])]

            ax.set_title('{} ({})'.format(pred_label, label),
                         color=('green' if pred_label == label else 'red'),
                         fontsize=15)
        else:
            img = np.transpose(np.clip(images[idx], 0, 1), (1, 2, 0))
            plt.imshow(img)
            ax.set_title(label, fontsize=15)


def hist(path='data/train', show_description=True, targets=None):
    df, folder_dict = get_folders(path)
    
    if show_description:
        if df["length"].nunique() == 1:
            print(f'There are {df["length"].unique()[0]} images in each folder')
        else:
            for name, length in folder_dict.items():
                print(f'There are {length} images in {targets[name]}')
    
    sns.displot(x='name', y='length', data=df).set(title='Distribution of Length in Folders')
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
    return pd.DataFrame({'name': folders_dict.keys(), 'length': folders_dict.values()}), folders_dict


def train_valid(train, valid):
    plt.title('Train/Validation Losses')
    plt.plot(np.arange(len(valid)), valid, label='Validation')
    plt.plot(np.arange(len(train)), train, label='Train')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def show_test_results(test_dict):
    test_results = pd.DataFrame({'name': test_dict.keys(), 'accuracy': test_dict.values()})
    test_results.sort_values(by='accuracy', ascending=False, inplace=True)

    first_half = test_results.iloc[:52, :]
    second_half = test_results.iloc[52:, :]
    fig = px.histogram(first_half.sort_values(by='accuracy', ascending=False),
                       x='name', y='accuracy', title=f'TOP Accuracy Distribution (%)')
    fig.show()

    fig = px.histogram(second_half.sort_values(by='accuracy', ascending=False),
                       x='name', y='accuracy', title=f'Bottom Accuracy Distribution (%)')
    fig.show()
