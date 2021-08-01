import os
import shutil

import re
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def create_folders(df):
    """
    This function creates three folders and its sub-folders starting from 1 to 102.
        :param df: Dataframes that store path and targets
    """
    # verify 'target' column exists
    assert 'target' in df.columns

    # create a list to iterate
    folders = ['train', 'test', 'valid']
    # create folders depending on their unique values
    for folder in folders:
        for n in df['target'].unique():
            directory = f'data/{folder}/{str(n)}'
            # make a folder only if it doesn't exist
            if not os.path.exists(directory):
                os.makedirs(directory)


def prettify(labels):
    """
    This function takes the target values read from "paths/class_names.txt" and removes unnecessary punctuation marks.
        :param labels: Target names (not numbers)
    :return:
    """
    # make sure there are 102 instances
    assert len(labels) == 102

    # convert labels dictionary
    labels_dict = labels.to_dict()['target']

    for k, v in labels_dict.items():
        v = re.findall('[a-zA-Z\s\-]+', v)[1].title()
        labels_dict[k] = v

    def inverse_mapping(f):
        return f.__class__(map(reversed, f.items()))

    return labels_dict, inverse_mapping(labels_dict)


def transfer(dfs: list, target):
    """
    This function sorts images to their respective folders.
        :param dfs: List of dictionaries (train, test, validation)
        :param target: Target labels of flowers
    """
    # make sure the folder that contains over 8000 images exist
    # if it doesn't, download here: https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz
    assert os.path.exists('jpg')

    # reset indexing by adding 1 (target values start from 1 in the MATLAB file)
    dfs[0].idx += 1
    dfs[1].idx += 1
    dfs[2].idx += 1

    # convert to dictionaries for easier lookup
    train_dict = dfs[0].set_index('path')['idx'].to_dict()
    test_dict = dfs[1].set_index('path')['idx'].to_dict()

    # read files from a folder and sort according to their location
    for k, v in target.set_index('path')['target'].to_dict().items():
        filename = f'jpg/{k}'
        # find the location of the image
        if filename in train_dict:
            to_path = f'data/train/{v}'
        elif filename in test_dict:
            to_path = f'data/test/{v}'
        else:
            to_path = f'data/valid/{v}'
        # transfer the image to its repsective folder
        shutil.move(filename, to_path.strip())


def create_loaders(n_batches, transformations=None):
    """
    This function applies transformations, sets batches, and creates loaders using PyTorch's ImageFolder and DataLoader.
        :param n_batches: Number of batches to use in DataLoader
        :param transformations: Presence of custom transformations
    :return: train, validation, and test loaders
    """
    # check if transformations exist
    if transformations is None:
        transformations = {
            # create transformations to use on training set
            'train': transforms.Compose([
                transforms.Resize(226),
                transforms.CenterCrop(224),
                transforms.RandomRotation(degrees=(-10, 10)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]),
            # create transformations to use on test and validation sets
            'test_valid': transforms.Compose([
                transforms.Resize(226),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        }

    # get the images from folders and apply appropriate transformations
    train_holder = datasets.ImageFolder('data/train', transform=transformations['train'])
    test_holder = datasets.ImageFolder('data/test', transform=transformations['test_valid'])
    valid_holder = datasets.ImageFolder('data/valid', transform=transformations['test_valid'])

    # define loaders
    train_loader = DataLoader(train_holder, batch_size=n_batches, shuffle=True)
    test_loader = DataLoader(test_holder, batch_size=n_batches, shuffle=True)
    valid_loader = DataLoader(valid_holder, batch_size=n_batches, shuffle=True)

    # return the three ready-to-go loaders
    return train_loader, test_loader, valid_loader