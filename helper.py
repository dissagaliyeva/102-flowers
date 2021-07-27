import os
import shutil

import re
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


def create_folders(df):
    """

    :param df:
    :return:
    """
    folders = ['train', 'test', 'valid']
    for folder in folders:
        for n in df['target'].unique():
            directory = f'data/{folder}/{str(n)}'
            if not os.path.exists(directory):
                os.makedirs(directory)


def prettify(labels):
    """

    :param labels:
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


def transfer(dfs: list, df):
    """
    This function sorts images to their respective folders.
    :param dfs: List of dictionaries (train, test, validation)
    :param labels: Target labels of flowers
    """
    # make sure the folder that contains over 8000 images exist
    # if it doesn't, download here: https://s3.amazonaws.com/fast-ai-imageclas/oxford-102-flowers.tgz
    assert os.path.exists('jpg')

    dfs[0].idx += 1
    dfs[1].idx += 1
    dfs[2].idx += 1

    # convert to dictionaries for easier lookup
    train_dict = dfs[0].set_index('path')['idx'].to_dict()
    test_dict = dfs[1].set_index('path')['idx'].to_dict()
    valid_dict = dfs[2].set_index('path')['idx'].to_dict()

    # read files from a folder and sort according to their location
    for k, v in df.set_index('path')['target'].to_dict().items():
        filename = f'jpg/{k}'

        if filename in train_dict:
            to_path = f'data/train/{v}'
        elif filename in test_dict:
            to_path = f'data/test/{v}'
        else:
            to_path = f'data/valid/{v}'

        shutil.move(filename, to_path.strip())


def create_loaders(n_batches):
    """

    :param n_batches:
    :return:
    """
    # create different transformation for train and test/validation sets
    transformations = {
        'train': transforms.Compose([
            transforms.Resize(226),
            transforms.CenterCrop(224),
            transforms.RandomRotation((-10, 10)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]),
        'test_valid': transforms.Compose([
            transforms.Resize(226),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    }

    # get the images from folders
    train_holder = datasets.ImageFolder('data/train', transform=transformations['train'])
    test_holder = datasets.ImageFolder('data/test', transform=transformations['test_valid'])
    valid_holder = datasets.ImageFolder('data/valid', transform=transformations['test_valid'])

    # define loaders
    train_loader = DataLoader(train_holder, batch_size=n_batches, shuffle=True)
    test_loader = DataLoader(test_holder, batch_size=n_batches, shuffle=True)
    valid_loader = DataLoader(valid_holder, batch_size=n_batches, shuffle=True)

    return train_loader, test_loader, valid_loader
