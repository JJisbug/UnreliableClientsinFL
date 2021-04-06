#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch import autograd
from tensorboardX import SummaryWriter
from scipy import stats
import math

from sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, server_test
from options import args_parser
from Update import LocalFSVGRUpdate, LocalUpdate, ServerUpdate,  LocalFedProxUpdate
from FedNets import MLP1, CNNMnist, CNNCifar
from averaging import   average_weights

import defence_mechanism

def test(net_g, data_loader, args):
    test_loss = 0
    correct = 0

    for idx, (data, target) in enumerate(data_loader): 
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        data, target = autograd.Variable(data), autograd.Variable(target)
        log_probs = net_g(data)
        test_loss += F.nll_loss(log_probs, target, size_average=False).data[0] # sum up batch loss
        y_pred = log_probs.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    return correct, test_loss

def calculate_avg_grad(users_g):
    avg_grad = np.zeros(users_g[0][1].shape)
    total_size = np.sum([u[0] for u in users_g]) #Total number of samples of all users
    for i in range(len(users_g)):
        avg_grad = np.add(avg_grad, users_g[i][0] * users_g[i][1])#+= users_g[i][0] * users_g[i][1]   <=> n_k * grad_k
    avg_grad = np.divide(avg_grad, total_size)#/= total_size
    # print("avg_grad:", avg_grad)
    return avg_grad

def traverseList(nestList): 
    flatList = []
    for item in nestList:
        if isinstance(item, list):
            flatList.extend(traverseList(item))
        else:
            flatList.append(item)
    return flatList

def flat(w):  
    a = []
    for this_key in w.keys():
        a.append(list(copy.deepcopy(w[this_key].flatten().cpu().numpy())))
    return traverseList(a)
    
np.set_printoptions(threshold=100000)   

if __name__ == '__main__':
    # parse args
    args = args_parser()
    algo_list = ['fedavg', 'fedprox', 'fsvgr']
    # define paths
    path_project = os.path.abspath('..')

    summary = SummaryWriter('local')
    args.gpu = -1          # -1 (CPU only) 
    args.lr = 0.002         # 0.001 for cifar dataset
    args.model = 'mlp'      # 'mlp' or 'cnn'
    args.dataset = 'mnist'  #  'cifar' or 'mnist'
    args.num_users = 5
    args.local_bs = 5 
    args.probability = 1  # Probability of malicious user attack
    args.global_test = 2000 
    
    args.malicious = True
    args.defense = False
    args.malicious_num = 1
    args.iter_num = 0
    args.q = 1
    args.iid = True   
    args.verbose = False  
    print("dataset:", args.dataset, " num_users:", args.num_users, "local_bs:", args.local_bs)
    if args.malicious:
        print("malicious_num", args.malicious_num, "iter_num", args.iter_num, "probability", args.probability)
          
    def Function(local_ep, T):
      # load dataset and split user
      dict_users = {}
      dataset_train = []
      if args.dataset == 'mnist':
          dataset_train = datasets.FashionMNIST('../data/fashionmnist/', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ]))
          dataset_test = datasets.FashionMNIST('../data/fashionmnist/', train=False, download=True,
                     transform=transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ]))
          # sample users
          if args.iid:
              dict_users = mnist_iid(dataset_train, args.num_users)
          else:
              dict_users = mnist_noniid(dataset_train, args.num_users)
      elif args.dataset == 'cifar':
          transform = transforms.Compose( 
             
              [transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
          dataset_train = datasets.CIFAR10('../data/cifar', train=True, transform=transform, target_transform=None, download=True)
          dataset_test = datasets.CIFAR10('../data/cifar', train=False, transform=transform, target_transform=None)
          if args.iid:
              dict_users = cifar_iid(dataset_train, args.num_users)
          else:
              dict_users = cifar_noniid(dataset_train, args.num_users)
              #exit('Error: only consider IID setting in CIFAR10')
      else:
          exit('Error: unrecognized dataset')
      img_size = dataset_train[0][0].shape
  
      # build model
      net_glob = None
      if args.model == 'cnn' and args.dataset == 'cifar':
          if args.gpu != -1:
              torch.cuda.set_device(args.gpu)
              net_glob = CNNCifar(args=args).cuda()
          else:
              net_glob = CNNCifar(args=args)
      elif args.model == 'cnn' and args.dataset == 'mnist':
          if args.gpu != -1:
              torch.cuda.set_device(args.gpu)
              net_glob = CNNMnist(args=args).cuda()
          else:
              net_glob = CNNMnist(args=args)
      elif args.model == 'mlp':
          len_in = 1
          for x in img_size:
              len_in *= x
          if args.gpu != -1:
              torch.cuda.set_device(args.gpu)
              # net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes).cuda()
              net_glob = MLP1(dim_in= len_in, dim_hidden=256, dim_out=args.num_classes).cuda()
          else:
              # net_glob = MLP1(dim_in=len_in, dim_hidden=128, dim_out=args.num_classes)
              net_glob = MLP1(dim_in=len_in, dim_hidden=256, dim_out=args.num_classes)
      else:
          exit('Error: unrecognized model')
      print("Nerual Net:",net_glob)
      net_glob.train()  #Train() does not change the weight values
      # copy weights
      w_glob = net_glob.state_dict()
  
      w_size = 0
      for k in w_glob.keys():
          size = w_glob[k].size()
          if(len(size)==1):
              nelements = size[0]
          else:
              nelements = size[0] * size[1]
          w_size += nelements*4
          # print("Size ", k, ": ",nelements*4)
      print("Weight Size:", w_size, " bytes")
      print("Weight & Grad Size:", w_size*2, " bytes")
      print("Each user Training size:", 784* 8/8* args.local_bs, " bytes")
      print("Total Training size:", 784 * 8 / 8 * 60000, " bytes")
      # training

      
      ### centralized training
      dataset_c_test = server_test(dataset_test, args.global_test)
      net_start = copy.deepcopy(net_glob)


    ###  FedAvg Aglorithm  ###
    
      epoch_acc = []
      epoch_loss = []
      server_acc = np.zeros(args.num_users, dtype='float')
      net_server = copy.deepcopy(net_start)
      net_glob = copy.deepcopy(net_start)
      dataset_users_test = server_test(dataset_test, 2000)
      global_test = copy.deepcopy(dataset_c_test)
      person = [[] for _ in range(args.num_users)]
      w_list = []
      
      epochs = int(math.ceil(T/local_ep))
      print('epochs:', epochs, 'local_ep:', local_ep)
      com = 0
      y11=[];y12=[]
      for iter in range(epochs):
          com += local_ep
          if com > T:     
              local_ep -= (com-T)
              com = 0
          epoch_server_acc = []
          w_locals, loss_locals, acc_locals = [], [], []
          for idx in range(args.num_users):
              local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], tb=summary)
              w, loss, acc = local.update_weights(net=copy.deepcopy(net_glob), local_ep=int(local_ep))
              pro = random.random()
              if args.malicious:                   
                  if iter >= args.iter_num and idx >= 0 and idx < args.malicious_num:
                      if pro < 1: 
                              for this_key in w.keys():
                                  dev = np.random.normal(0, 0.2, w[this_key].size())
                                  dev = torch.from_numpy(dev).float()
                                  w[this_key] = w[this_key] * (-0.5)+dev

                      else:
                          if pro < 0:   
                              for this_key in w.keys():
                                  '''
                                  dev = np.random.normal(0, 0.5, w[this_key].size())
                                  dev = torch.from_numpy(dev).float().cuda()
                                  w[this_key] = w[this_key] + dev
                                  '''
                                  w[this_key] = w[this_key] * (0.5)   
                             
              w_locals.append(copy.deepcopy(w))
              loss_locals.append(copy.deepcopy(loss))
              acc_locals.append(copy.deepcopy(acc))   #train acc
              #print("User ", idx, " Acc:", acc, " Loss:", loss)
              
          for idx in range(args.num_users):
              net_server.load_state_dict(w_locals[idx])
              server = ServerUpdate(args=args, dataset=dataset_test, idxs=dataset_users_test, tb=summary)
              acc, loss = server.test(net=net_server) # test acc
              server_acc[idx] += acc
              epoch_server_acc.append(acc)
              #print(server_acc)
              #print(epoch_server_acc)
              
          #co-Defense start
          
          weight = copy.deepcopy(epoch_server_acc)
          for i in range(len(weight)):
              weight[i] = 1
          total = sum(weight)   
          for i in range(len(weight)):
              weight[i] /= total    #same weight average
          
          for i in range(len(w_locals)):
              if iter == 0:
                  w_list.append(copy.deepcopy(flat(w_locals[i])));
              else: 
                  a , _= stats.pearsonr(w_list[i], flat(w_locals[i]))  # -1~1  
                  
                  #w_list = np.array(w_list);w_locals = np.array(w_locals)
                  #a = np.linalg.norm(w_list[i]- flat(w_locals[i]))
                  #a = 1/(a+1)
                  
                  person[i].append(abs(a))
                  w_list[i] = copy.deepcopy(flat(w_locals[i]))
                  #print(person[i][-1])

          # update global weights
          if args.defense:
              if iter > 0:
                  for i in range(len(weight)):
                      weight[i] = copy.deepcopy(person[i][-1])
              total = sum(weight)
              for i in range(len(weight)):
                  weight[i] /= total    #weight turn to 1
                 
          w_glob = average_weights(w_locals, weight)
         
          #w_glob = defence_mechanism.defence_reweight_former(args, w_locals, epoch_server_acc)
          
          #w_glob = defence_mechanism.defence_SecProbe(args, w_locals, epoch_server_acc)
          
          #w_glob = defence_mechanism.defence_SecProbe_1(args, w_locals, epoch_server_acc)

          #w_glob = defence_mechanism.defence_Krum(args, w_locals, 10)
          
          # copy weight to net_glob
          net_glob.load_state_dict(w_glob)

          # global test
          net_local = ServerUpdate(args=args, dataset=dataset_test, idxs=global_test, tb=summary)
          return_acc, return_loss = net_local.test(net=net_glob)
          print("\nEpoch: {}, Global test loss {}, Global test acc: {:.2f}%".format(iter, return_loss, 100*return_acc))
    
          y11.insert(iter,return_loss)
          y12.insert(iter,return_acc) 
          
          # print loss
          
          if iter <= 200:
              if iter % 1 == 0 or iter == 0:
                  epoch_acc.append(return_acc)
                  epoch_loss.append(return_loss)
          else:
              if iter % 1 == 0:
                  epoch_acc.append(return_acc)
                  epoch_loss.append(return_loss)
          #if args.epochs % 1 == 0:
              #print('\nUsers Train Average loss:', loss_avg)
              #print('\nTrain Train Average accuracy', acc_avg)
          # if (return_average_acc > 0.89):
          #     break
      #for i in range(len(person)):
          #print(person[i])
      a = epoch_acc[-1]
      b = epoch_loss[-1]
      return a, b,y11,y12
    
    '''
    T = 1000
    acc_d = []
    loss_d = []
    for i in range(10):
        a, b = Function(100,T)
        acc_d.append(a)
        loss_d.append(b)
    print(acc_d, loss_d)
    print(sum(acc_d)/len(acc_d))
    print(sum(loss_d)/len(loss_d))
    '''

    a = []
    b = []
    loss_y = [0]*8
    acc_y = [0]*8
    times = 1
    for j in range(times):
      print("times = ",j)
      acc_d = []
      loss_d = [] 
      T = 16
      lo = [20]
      print("local_ep:",lo)
      for i in lo:
          acc, loss, y1, y2 = Function(i, T)
          acc_d.append(acc)
          loss_d.append(loss)
      a.append(acc_d)
      b.append(loss_d)
      #print(acc_d, loss_d)
      
      y1=np.array(y1);  y2=np.array(y2)
      loss_y=np.array(loss_y);  acc_y=np.array(acc_y)
      loss_y = loss_y + y1
      acc_y = acc_y + y2
      
    tt = np.array(copy.deepcopy(a[0]))
    vv = np.array(copy.deepcopy(b[0]))
    for i in range(times-1):
      tt += np.array(a[i+1])
      vv += np.array(b[i+1])
    tt = [x/times for x in tt]
    vv = [y/times for y in vv]
    print(tt,vv) #last epoch average all times

print("average loss: ",list(loss_y/times))
print("average acc: ",list(acc_y/times))