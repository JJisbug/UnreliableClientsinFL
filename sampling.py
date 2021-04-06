#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
import copy
from torchvision import datasets, transforms

# np.random.seed(1)

def unique_index(L,f):
    return [i for (i,value) in enumerate(L) if value==f]

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, 512, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        #dict_users[i] = all_idxs[10*(i+0):10*(i+1)]
    '''
    dict_users = {}
    labels = dataset.train_labels.numpy()
    classes = np.unique(labels)
    print(classes)
    classes_index = []
    for i in range(len(classes)):
        classes_index.append(unique_index(labels, classes[i]))
    for i in range(num_users):
        dict_users[i] = []
        for j in range(len(classes)):
            temp = np.random.choice(classes_index[j], 1, replace=False)
            classes_index[j] = list(set(classes_index[j]) - set(copy.deepcopy(temp)))
            dict_users[i].append(copy.deepcopy(temp))
        dict_users[i] = np.array(dict_users[i]).flatten()
    '''
    return dict_users
    

def server_test(dataset, num):
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    #dict_users = set(np.random.choice(all_idxs, num, replace=False))
    dict_users = all_idxs[1000:1000+num]
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    '''
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # print("total_data:",len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#Sort with labels
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        print(rand_set)
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users
    '''
    dict_users = {}
    labels = dataset.train_labels.numpy()
    classes = np.unique(labels)
    print(classes)
    classes_index = []
    for i in range(len(classes)):
        classes_index.append(unique_index(labels, classes[i]))
    for i in range(len(classes_index)):
        print(len(classes_index[i]))
    for i in range(num_users):
        dict_users[i] = np.random.choice(classes_index[i%10], 10, replace=False)
        classes_index[i%10] = list(set(classes_index[i%10]) - set(copy.deepcopy(dict_users[i])))
    return dict_users 
       

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# Divide into 100 portions of total data. Allocate 2 random portions for each user
def cifar_noniid(dataset, num_users):
    num_shards, num_imgs = 100, 500
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.train_labels)#.numpy()
    print(len(idxs))
    print(len(labels))
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 4, replace=False))
        #idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            #np.random.shuffle(dict_users[i])
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)