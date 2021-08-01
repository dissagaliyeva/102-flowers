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
    This function visualizes N batch images from loaders and applies de-normalization.
        :param iterable:     Iterable loader
        :param dictionary:   The dictionary that stores number:category values
        :param n_batch:      Number of images to have in the DataLoader
        :param actual_order: The true order of folders
        :param cuda:         Whether to run on GPU or CPU (default None)
        :param model:        The model to use (default None)
    """
    # create a transformation to de-normalize the images
    invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
                                   transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                        std=[1., 1., 1.]), ])
    # get next batch of images
    images, targets = next(iterable)

    # if GPU is on, transfer the images to that state
    if cuda:
        images = images.cuda()

    # if model is specified, predict the test image(-s)
    if model is not None:
        # get predictions
        output = model(images)
        # take only biggest probability
        _, pred = torch.max(output, 1)
        # transform it to numpy
        pred = np.squeeze(pred.numpy()) if not cuda else np.squeeze(pred.cpu().numpy())

    # apply de-normalization transformations
    images = invTrans(images)
    # convert to numpy
    images = images.numpy()

    # plot the images in the batch, along with the corresponding labels
    # set the figure size
    fig = plt.figure(figsize=(40, 10))
    # set the title
    fig.suptitle(f'Batch of {n_batch}', fontsize=25)
    # plot specified number of images
    for idx in np.arange(n_batch):
        # create subplots
        ax = fig.add_subplot(2, 10, idx + 1, xticks=[], yticks=[])
        # get the actual class name of the image
        label = dictionary[int(actual_order[targets[idx]])]

        # transform to CPU
        if model is not None:
            if cuda:
                images[idx] = images[idx].cpu()
            # transform the image
            img = np.transpose(np.clip(images[idx], 0, 1), (1, 2, 0))
            plt.imshow(img)
            # get the label
            pred_label = dictionary[int(actual_order[pred[idx]])]
            # show if it's a correct prediction in green, red otherwise
            ax.set_title('{} ({})'.format(pred_label, label),
                         color=('green' if pred_label == label else 'red'),
                         fontsize=15)
        else:
            # if it's not a prediction, show a batch of images
            img = np.transpose(np.clip(images[idx], 0, 1), (1, 2, 0))
            plt.imshow(img)
            # set the title
            ax.set_title(label, fontsize=15)


def hist(path='data/train', show_description=True, targets=None):
    """
    This function shows the distribution of images in each class. It mainly checks for class-imbalance.
        :param path:                Path to the folder that needs to be checked
        :param show_description:    Whether to show text representation along with image distribution
        :param targets:             Specific folder name to check
    """
    # get the contents of a folder in dataframe representation (name:folder size)
    df, folder_dict = get_folders(path)
    # set the variable that checks if there is imbalance
    different = True

    # text representation of contents
    if show_description:
        # check if it has a uniform distribution
        if df["length"].nunique() == 1:
            # set imbalance to False
            different = False
            # show the results
            print(f'There are {df["length"].unique()[0]} images in each folder')
        else:
            # if there is class imbalance in image folders, show how many images in each sub-folder
            for name, length in folder_dict.items():
                # print contents
                print(f'There are {length} images in {targets[name]}')

    # show histogram representation for easier information grasp
    if different:
        sns.displot(x='name', y='length', data=df).set(title='Distribution of Length in Folders')
        # remove unnecessary ticks
        plt.tick_params(
            axis='x',
            which='both',
            bottom=False,
            labelbottom=False
        )
        # set x label
        plt.xlabel('folders')
        # show the histogram
        plt.show()


def get_folders(path):
    """
    This function creates the easier representation of folders and their respective images
        :param path: Path to a specific folder
    :return: Dataframe that stores names and folder sizes
    """
    # verify the folder exists
    assert os.path.exists(path)

    # create an empty dictionary
    folders_dict = {}
    # walk through all folders in a path
    for folder in os.listdir(path):
        # store folder sizes
        folders_dict[folder] = len(os.listdir(path + '/' + folder))
    # return prettified representation
    return pd.DataFrame({'name': folders_dict.keys(), 'length': folders_dict.values()}), folders_dict


def train_valid(train: list, valid: list):
    """
    This function is used in the training/validation step. It shows the change of losses throughout N epochs.
        :param train: List of train losses
        :param valid: List of validation losses
    """
    # double-verification that both train and validation are lists
    assert type(train) == list and type(valid) == list

    # set title
    plt.title('Train/Validation Losses')
    # plot validation losses
    plt.plot(np.arange(len(valid)), valid, label='Validation')
    # plot train losses
    plt.plot(np.arange(len(train)), train, label='Train')
    # set labels
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # show labels
    plt.legend()
    # show plot
    plt.show()


def show_test_results(test_dict: dict):
    """
    This function is used in the test results through histogram. It sorts the results and show top best and worst
    predictions in the descending order.
        :param test_dict: Test results that stores 102 predictions
    :return: Returns best and worst results
    """
    # verify the results have 102 values
    assert len(test_dict) == 102

    # create a dataframe to store results
    test_results = pd.DataFrame({'name': test_dict.keys(), 'accuracy': test_dict.values()})
    # sort in the descending order
    test_results.sort_values(by='accuracy', ascending=False, inplace=True)

    # split the dataframe in half to show best and worst results
    first_half = test_results.iloc[:52, :]
    second_half = test_results.iloc[52:, :]
    # show best results
    fig = px.histogram(first_half.sort_values(by='accuracy', ascending=False),
                       x='name', y='accuracy', title=f'TOP Accuracy Distribution (%)')
    fig.show()

    # show worst results
    fig = px.histogram(second_half.sort_values(by='accuracy', ascending=False),
                       x='name', y='accuracy', title=f'Bottom Accuracy Distribution (%)')
    fig.show()
    return first_half, second_half


def side_by_side(results: list, n_plots: int, names: list):
    """
    This function shows side-by-side visualization of N training results. It shows both training and validation losses.
        :param results: List of final results of a specific model
        :param n_plots: Number of plots to show
        :param names:   List of names to use for titles
    """
    # verify there are equal number of results, plots, and names
    assert len(results) == len(names) == n_plots

    # create x axis
    x_axis = np.arange(1, len(results[0]['train_loss']) + 1)
    # get titles
    titles = [names[x] for x in range(n_plots)]

    # create the main placeholder for plots
    fig = make_subplots(rows=1, cols=n_plots, shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=titles)
    # iterate over each result and show on the same row
    for idx in range(n_plots):
        # plot train losses
        fig.add_trace(go.Scatter(x=x_axis, y=results[idx]['train_loss'],
                                 marker=dict(color='Blue'), name=f'{names[idx]} train'), row=1, col=idx + 1)
        # plot validation losses
        fig.add_trace(go.Scatter(x=x_axis, y=results[idx]['valid_loss'],
                                 marker=dict(color='Red'), name=f'{names[idx]} valid'), row=1, col=idx + 1)
    # show the whole figure
    fig.show()


def visualize_most_confused(target: str, confused_dict: dict, label_inverse: dict, top_k=3):
    """
    This function visualizes a class/classes that were mainly confused. It shows the total number of flowers the model
    confused it with. Additionally, it shows its confusions.
        :param target:          Class name (string) to check
        :param confused_dict:   Dictionary of confusions (declared in the training/validation/test function)
        :param label_inverse:   Dictionary that stores class_name:index
        :param top_k:           Number of confusions to show
    """
    # verify the target is a string
    assert type(target) == str

    # create a function to read a random image from a test folder
    paths = lambda x: get_image(label_inverse[x])

    # create a variable that stores all confusions to sort and visualize them
    temp = pd.DataFrame({'name': confused_dict[target].keys(), 'confused': confused_dict[target].values()})
    # get the paths to a random image
    temp['paths'] = temp.name.apply(paths)

    # sort the confusions in the descending order
    temp.sort_values(by='confused', ascending=False, inplace=True)

    # visualize top confused classes (excluding the target itself)
    fig = px.histogram(temp, x='name', y='confused', title=f'Confused {target} with')
    fig.show()

    # get images' paths to visualize
    target_path = paths(target)
    top_paths = temp.iloc[:top_k, :]
    top_k = len(top_paths)

    # visualize the target first
    plt.title(f'Showing {target} with TOP {top_k} confusions')
    plt.imshow(Image.open(target_path))
    plt.show()
    # visualize target's top confused classes
    fig = plt.figure(figsize=(10, 10))
    for idx in range(top_k):
        # show each image individually
        ax = fig.add_subplot(1, top_k + 1, idx + 1, xticks=[], yticks=[])
        plt.imshow(Image.open(top_paths.iloc[idx, 2]))
        # show its class name
        ax.set_title(top_paths.iloc[idx, 0])
    plt.show()


def get_image(idx):
    """
    This function selects a random image from test folder.
        :param idx: Class folder to read from
    :return: A random image from a specific folder
    """
    return random.choice([f'data/test/{idx}/{x}' for x in os.listdir(f'data/test/{idx}')])


def get_confusions(confused: dict, label_dict: dict, actual_order: list):
    """
    This function takes in all the confusions made in the test mode (each item in the dictionary has at least
    102 values), counts each class, and stores the information in an accessible manner. Its main purpose is to
    find the most confused instances while not taking its own class name in the account.
        :param confused:        Dictionary of confusions (integers only, without class names)
        :param label_dict:      Dictionary that stores index:class_name values
        :param actual_order:    List of true order of folders
    :return: Logically structures dictionary with class names and its dictionary of confusions with the occurrances
    """
    # create an empty dictionary to store corrected values
    corrected = {}
    # iterate over the whole messy structure
    for k, v in confused.items():
        for val in v:
            # check if class name exists (to remove class name itself from confusions)
            if val == int(k): continue
            # otherwise, get their actual names
            l1 = get_item(int(k), label_dict, actual_order)
            l2 = get_item(val, label_dict, actual_order)
            # if it exists, add one more instance
            if l1 in corrected:
                if l2 in corrected[l1]:
                    corrected[l1][l2] += 1
                else:
                    # otherwise, create a new instance
                    corrected[l1][l2] = 0
            else:
                # if the confused class name is not in the dictionary, create a new dictionary to store them
                corrected[l1] = {}
    return corrected


def get_item(idx, label_dict: dict, actual_order: list):
    """
    This function returns the true class name of an item at specific index.
        :param idx:           Index of a class
        :param label_dict:    Dictionary that stores index:class_name values
        :param actual_order:  List of true order of folders
    :return: Class name
    """
    return label_dict[int(actual_order[idx])]
