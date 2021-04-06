# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:10:28 2019

@author: WEIKANG
"""

import numpy as np
import copy
import torch
import random
import time
import math
from Calculate import get_2_norm, average_weights

def defence_reweight_acc(args, w, list_acc_detect):
    
    w_avg = copy.deepcopy(w[0])
    avg_acc = sum(list_acc_detect)
    acc_detect = copy.deepcopy(list_acc_detect/avg_acc)
    p_w = copy.deepcopy(acc_detect)
    print("\nNew weights", p_w)   
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += p_w[i]*w[i][k]
       
    return w_avg

def defence_reweight_former(args, w, list_acc_detect):
    
    w_avg = copy.deepcopy(w[0])
    avg_acc = sum(list_acc_detect)/len(list_acc_detect)
    acc_detect = copy.deepcopy(list_acc_detect/avg_acc)
    p_w = copy.deepcopy(acc_detect)
    m0, m1, m2, m3, m4, m5= 0, 0, 0, 0, 0, 0
    for j in range(len(acc_detect)):
        if acc_detect[j] <= 0.7:
            p_w[j] = copy.deepcopy(1-np.exp(-args.q/10))
            m0 += 1
        elif (acc_detect[j] <= 0.75) and (acc_detect[j] > 0.7):
            p_w[j] = copy.deepcopy(1-np.exp(-args.q/5))
            m1 += 1 
        elif (acc_detect[j] <= 0.8) and (acc_detect[j] > 0.75):
            p_w[j] = copy.deepcopy(1-np.exp(-args.q/4)) 
            m2 += 1 
        elif (acc_detect[j] <= 0.85) and (acc_detect[j] > 0.8):
            p_w[j] = copy.deepcopy(1-np.exp(-args.q/3))
            m3 += 1   
        elif (acc_detect[j] <= 0.9) and (acc_detect[j] > 0.85):
            p_w[j] = copy.deepcopy(1-np.exp(-args.q/2))
            m4 += 1               
        elif (acc_detect[j] <= 0.95) and (acc_detect[j] > 0.9):
            p_w[j] = copy.deepcopy(1-np.exp(-args.q))
            m5 += 1                
    for j in range(len(acc_detect)):            
        if acc_detect[j] > 0.95:
            p_w[j] = copy.deepcopy(1+(m0*np.exp(-args.q/10)+m1*np.exp(-args.q/5)+\
               m2*np.exp(-args.q/4)+m3*np.exp(-args.q/3)+m4*np.exp(-args.q/2)+\
            m5*np.exp(-args.q))/(len(acc_detect)-(m0+m1+m2+m3+m4+m5)))  
    print("\nNew weights", p_w/len(w))     
    if sum(p_w) != len(p_w):
       print("\nError weights", sum(p_w)/len(p_w)) 
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += p_w[i]*w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def defence_SecProbe(args, w, list_acc_detect):
    
    w_avg = copy.deepcopy(w[0])
    avg_acc = sum(list_acc_detect)/len(list_acc_detect)
    acc_detect = copy.deepcopy(list_acc_detect/avg_acc)
    p_w = copy.deepcopy(acc_detect)
    m2=0
    for j in range(len(acc_detect)):
        m1 = 0
        for i in range(len(acc_detect)):
            if acc_detect[j] < acc_detect[i]:
                m1 += 1;
                if m1 >= (math.floor(len(acc_detect)/2)):
                    p_w[j]=0
    for n in range(len(acc_detect)):
        if p_w[n]!=0:
            m2 += 1; p_w[n] = 1;
    p_w = np.array(p_w)/m2
    p_w = list(p_w)
    print("\nNew weights", p_w) 
    for k in w_avg.keys():
        for i in range(0, len(w)):
             if i == 0:
                w_avg[k] *= p_w[i]
             else:
                w_avg[k] += p_w[i]*w[i][k]
        #w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg



def defence_neur_net(args, w, d_out):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            if d_out[i]==1:
                w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], sum(d_out))
    return w_avg

def defence_Krum(args, w, c):
    c = c+1
    euclid_dist_list = []
    euclid_dist_matrix = [[0 for i in range(len(w))] for j in range(len(w))]
    for i in range(len(w)):
        for j in range(i,len(w)):
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
        euclid_dist = euclid_dist_matrix[i][:]
        euclid_dist.sort()
        if len(w)>=c:
            euclid_dist_list.append(sum(euclid_dist[:c]))
        else:
            euclid_dist_list.append(sum(euclid_dist))        
    # euclid_dist_list = []
    # for i in range(len(w)):
    #     euclid_dist = []
    #     for j in range(len(w)):
    #         if j != i:
    #             euclid_dist.append(get_2_norm(w[j],w[i]))
    #     euclid_dist.sort()
    #     if len(w)>=c:
    #         euclid_dist_list.append(sum(euclid_dist[:c]))
    #     else:
    #         euclid_dist_list.append(sum(euclid_dist))

    s_w = euclid_dist_list.index(min(euclid_dist_list))    
    w_avg = w[s_w]
    return  w_avg  #s_w, euclid_dist_list, euclid_dist_matrix,

def defence_Krum_dynamic(args,w,c,id_malicious,euclid_dist_matrix):
    c = c+1
    for i in id_malicious:
        for j in range(len(id_malicious),len(w)):        
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
            
    euclid_dist_list = []
    for i in range(len(w)):       
        euclid_dist = euclid_dist_matrix[i][:]
        euclid_dist.sort()
        if len(w)>=c:
            euclid_dist_list.append(sum(euclid_dist[:c]))
        else:
            euclid_dist_list.append(sum(euclid_dist))
            
    s_w = euclid_dist_list.index(min(euclid_dist_list))
    w_avg = w[s_w]
    return s_w, euclid_dist_list,euclid_dist_matrix, w_avg

def defence_Krum_par(args,w,c,id_malicious,euclid_dist_matrix):
    c = c+1
    for i in id_malicious:
        for j in range(len(id_malicious),len(w)):        
            euclid_dist_matrix[i][j] = get_2_norm(w[i],w[j])
            euclid_dist_matrix[j][i] = euclid_dist_matrix[i][j]
            
    euclid_dist_list = []
    for i in range(len(w)):  
        if i < len(id_malicious):
            euclid_dist = euclid_dist_matrix[i][:]
            euclid_dist.sort()
            if len(w)>=c:
                euclid_dist_list.append(sum(euclid_dist[:c]))
            else:
                euclid_dist_list.append(sum(euclid_dist))
        else:
            euclid_dist = euclid_dist_matrix[i][:]
            euclid_di = euclid_dist_matrix[i][len(id_malicious):]
            for j in id_malicious:
                euclid_dist[j] = copy.deepcopy((1-args.r)*min(euclid_di)+args.r*max(euclid_di))
            euclid_dist.sort()
            if len(w)>=c:
                euclid_dist_list.append(sum(euclid_dist[:c]))
            else:
                euclid_dist_list.append(sum(euclid_dist))            
                
    s_w = euclid_dist_list.index(min(euclid_dist_list))
    w_avg = w[s_w]
    return s_w, euclid_dist_list,euclid_dist_matrix, w_avg

def defence_Trimmed_mean(args, w, beta):
    if isinstance(w[0],dict):
        w_avg = average_weights(w)
        for k in w_avg.keys():
            for index, element in np.ndenumerate(w[0][k].cpu().numpy()):
                max_record = []
                for i in range(beta):
                    max_record.append(copy.deepcopy(w[i][k][index]))
                min_record = copy.deepcopy(max_record)
                max_record.sort(reverse=True)
                min_record.sort()
                for i in range(beta, len(w)):
                    elem = copy.deepcopy(w[i][k][index])
                    for j in range(len(max_record)):
                        if elem > max_record[j]:
                            max_record[j] = copy.deepcopy(elem)
                            break
                    for j  in range(len(min_record)):
                        if elem < min_record[j]:
                            min_record[j] = copy.deepcopy(elem)
                            break
                w_avg[k][index] = copy.deepcopy(len(w)*w_avg[k][index]/(len(w)-2*beta)\
                     -sum(max_record)/(len(w)-2*beta)-sum(min_record)/(len(w)-2*beta))
    else:
        w_avg = average_weights(w)
        for index, element in np.ndenumerate(w[0]):
            max_record = []
            for i in range(beta):
                max_record.append(copy.deepcopy(w[i][index]))
            min_record = copy.deepcopy(max_record)
            max_record.sort(reverse=True)
            min_record.sort()
            for i in range(beta, len(w)):
                elem = copy.deepcopy(w[i][index])
                for j in range(len(max_record)):
                    if elem > max_record[j]:
                        max_record[j] = copy.deepcopy(elem)
                        break
                for j  in range(len(min_record)):
                    if elem < min_record[j]:
                        min_record[j] = copy.deepcopy(elem)
                        break
            w_avg[index] = copy.deepcopy(len(w)*w_avg[index]/(len(w)-2*beta)\
                 -sum(max_record)/(len(w)-2*beta)-sum(min_record)/(len(w)-2*beta))
    return w_avg 