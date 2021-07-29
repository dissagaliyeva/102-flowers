import torch
from torch import nn
from torch import optim

import helper
import visualize
import numpy as np


def set_params(n_epochs: int, model, label_dict, actual_order,
               use_cuda: bool, save_path: str, learning_decay=False,
               criterion_name='CrossEntropy', optim_name='SGD', n_batch=20):
    print(f'''========== Starting Training ==========
    Loss function: {criterion_name}
    Optimizer: {optim_name}
    Batch size: {n_batch}
    Path: {save_path}
    ''')

    train_loader, test_loader, valid_loader = helper.create_loaders(n_batch)
    loaders = {'train': train_loader, 'test': test_loader, 'valid': valid_loader}

    criterion = nn.CrossEntropyLoss()
    optimizer = None
    lr_decay = None
    if optim_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    elif optim_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=1e-4)

    if learning_decay:
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

    model, train_loss, valid_loss = train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path, lr_decay)
    visualize.train_valid(train_loss, valid_loss)

    confused_with, test_dict, test_loss = test(loaders, model, criterion, use_cuda, label_dict, actual_order)
    # visualize.show_test_results(test_dict)

    print(f'''========== Ending Training ==========
    Train loss: {train_loss[-1]}
    Valid loss: {valid_loss[-1]}
    Test  loss: {test_loss}
    ''')

    return [train_loss, valid_loss, test_loss, model, confused_with, test_dict]


def train(n_epochs: int, loaders: dict, model, optimizer,
          criterion, use_cuda: bool, save_path: str, learning_decay):
    """
        This function trains the model and shows the progress.

        Parameters:
            n_epochs (int): Number of epochs to train for
            loaders (dict): Dictionary of loaders to use
            model: Model being used
            optimizer: Selected optimizer
            criterion: Loss function
            use_cuda (bool): If GPU is enables or not
            save_path (str): Path to store the results in
            learning_decay: Learning rate decay scheduler to use

        Returns:
            A trained model, train and validation losses
    """
    train_losses = []
    valid_losses = []

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs + 1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        # set the module to training mode
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # record the average training loss, using something like
            optimizer.zero_grad()

            # get the final outputs
            output = model(data)

            # calculate the loss
            loss = criterion(output, target)

            # start back propagation
            loss.backward()

            # update the weights
            optimizer.step()

            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        # set the model to evaluation mode
        model.eval()
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()

            # update average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.size(0)

        train_loss /= len(loaders['train'].sampler)
        valid_loss /= len(loaders['valid'].sampler)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics every 5 epochs
        if epoch % 5 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch,
                train_loss,
                valid_loss
            ))

        # if the validation loss has decreased, save the model at the filepath stored in save_path
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss

        # update learning rate decay
        if learning_decay:
            learning_decay.step()

    return model, train_losses, valid_losses


def test(loaders, model, criterion, use_cuda, label_dict, actual_order):
    """
    This functions calculates the correctness and shows the results of the architecture.

    Parameters:
        loaders: Dictionary that stores all three loaders
        model: Model used for implementation
        criterion: Loss function
        use_cuda: If GPU is available or not

    Returns:
        The accuracy of the model
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # keep track of correctly classified classes
    class_correct = list(0. for _ in range(102))
    class_total = list(0. for _ in range(102))

    test_dict = {}
    confused_with = {}

    # set the module to evaluation mode
    model.eval()

    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)

        # calculate the loss
        loss = criterion(output, target)

        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))

        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]

        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

        for i in range(len(target)):
            label = target.data[i]

            if int(label) not in confused_with:
                confused_with[int(label)] = []
            else:
                confused_with[int(label)].append(pred[i].item())

            class_correct[label] += np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy()[i].item()
            class_total[label] += 1
    # show the accuracy
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    for i in range(102):
        if class_total[i] > 0:
            name = label_dict[int(actual_order[i])]
            accuracy = 100 * class_correct[i] / class_total[i]
            # print(f'Test Accuracy of {name}: %{accuracy}'
            #       f'({np.sum(class_correct[i])}/{np.sum(class_total[i])})')
            test_dict[name] = accuracy

    return confused_with, test_dict, test_loss
