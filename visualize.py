import json

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image
import random

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
    for idx in np.arange(n_batch):
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
    plt.legend()
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
    return first_half, second_half


def side_by_side(results: list, n_plots, names):
    x_axis = np.arange(len(results[0]['train_loss']))
    titles = [names[x] for x in range(n_plots)]

    fig = make_subplots(rows=1, cols=n_plots, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=titles)
    for idx in range(n_plots):
        fig.add_trace(go.Scatter(x=x_axis, y=results[idx]['train_loss'],
                                 marker=dict(color='Blue'), name=f'{names[idx]} train'), row=1, col=idx+1)
        fig.add_trace(go.Scatter(x=x_axis, y=results[idx]['valid_loss'],
                                 marker=dict(color='Red'), name=f'{names[idx]} valid'), row=1, col=idx+1)
    fig.show()


def visualize_most_confused(target, confused_dict=None, top_k=3, label_inverse=None):
    paths = lambda x: get_image(label_inverse[x])

    temp = pd.DataFrame({'name': confused_dict[target].keys(), 'confused': confused_dict[target].values()})
    temp['paths'] = temp.name.apply(paths)

    temp.sort_values(by='confused', ascending=False, inplace=True)

    fig = px.histogram(temp, x='name', y='confused', title=f'Confused {target} with')
    fig.show()

    target_path = paths(target)
    top_paths = temp.iloc[:top_k, :]
    top_k = len(top_paths)

    plt.title(f'Showing {target} with TOP {top_k} confusions')
    plt.imshow(Image.open(target_path))
    plt.show()

    fig = plt.figure(figsize=(10, 10))
    for idx in range(top_k):
        ax = fig.add_subplot(1, top_k + 1, idx + 1, xticks=[], yticks=[])
        plt.imshow(Image.open(top_paths.iloc[idx, 2]))
        ax.set_title(top_paths.iloc[idx, 0])
    plt.show()


def get_image(idx):
    return random.choice([f'data/train/{idx}/{x}' for x in os.listdir(f'data/train/{idx}')])


def get_confusions(confused, label_dict, actual_order):
    corrected = {}
    for k, v in confused.items():
        for val in v:
            if val == int(k): continue
            l1 = get_item(int(k), label_dict, actual_order)
            l2 = get_item(val, label_dict, actual_order)

            if l1 in corrected:
                if l2 in corrected[l1]:
                    corrected[l1][l2] += 1
                else:
                    corrected[l1][l2] = 0
            else:
                corrected[l1] = {}
    return corrected


def get_item(idx, label_dict, actual_order):
    return label_dict[int(actual_order[idx])]