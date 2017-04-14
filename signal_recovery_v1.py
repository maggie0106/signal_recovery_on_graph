#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 09 12:23:25 2017

@author: YingxueZhang
"""
import networkx as nx
import numpy as np  
import matplotlib.pyplot as plt
import pandas as pd
import sys

#graph generation
def graph_generation(N,graph_type,base_mode):
    if graph_type=='ring graph':
        k=4   #neighbor number
        p=0
        G=nx.newman_watts_strogatz_graph(N, k, p, seed=None)    #ring graph
        fourier_basis=normalized_fourier_basis(G,base_mode)
    elif graph_type=='random graph':
        G = nx.erdos_renyi_graph(N,0.1)                         #random graph(Erdos-Renyi Graph)
        fourier_basis=normalized_fourier_basis(G,base_mode)
    elif graph_type=='random geometric graph':
        G=nx.random_geometric_graph(N,0.1)
        fourier_basis=normalized_fourier_basis(G,base_mode)
    else:
        print 'this graph type is not allowed'
        sys.exit()
    return fourier_basis

def matrix_eigenvalue_normalization(M):
    Sigma_M, V_M = np.linalg.eigh(M) 
    index_sig = np.argsort(Sigma_M)
    Sigma_M = Sigma_M[index_sig[::-1]] #sorting the eigenvalue
    V_M = V_M[:,index_sig[::-1]] #sorting the according eigenvectors
    #Normalized matrix A
    M=M/Sigma_M[0] #normalized matrix A in order to let the biggest eigenvalue=1
    Sigma_M, V_M = np.linalg.eigh(M) 
    index_sig = np.argsort(Sigma_M)
    Sigma_M = Sigma_M[index_sig[::-1]]
    Normalize_eigenvector = V_M[:,index_sig[::-1]]
    return  Normalize_eigenvector
    
#fourier basis generation
def normalized_fourier_basis(G,base_mode):
    if base_mode=='adjacency':
        A=nx.adjacency_matrix(G, nodelist=None, weight='weight').todense() 
        Fourier_basis=matrix_eigenvalue_normalization(A)
        print 'adjacency mode'
    elif base_mode=='Laplacian':
        Laplacian = nx.normalized_laplacian_matrix(G).todense()
        Fourier_basis=matrix_eigenvalue_normalization(Laplacian)
        print 'Laplacian mode'
    else:
        print 'this fourier_basis is not allowed'
        sys.exit()
    return Fourier_basis
    
#signal generation
def graph_signal_generation(n,K,beta,Fourier_basis):
    N=n
    K=10
    signal=[]
    for i in range(N):
        if i<K:
            signal.append(np.random.normal(1,0.5))
        else:
            #signal.append((float(K)/i)**(2*beta))
            signal.append(0)  #test the completely band limited case
    signal=np.asarray(signal)
    signal_norm=np.linalg.norm(signal)  #fourier transform x_f=U_A*x
    signal_norm=signal/signal_norm
    signal_norm=np.transpose(signal_norm)
    signal_norm=signal_norm.reshape(len(signal),1)

    x=Fourier_basis*signal_norm
    x=np.asarray(x)
    x=x.reshape(n,1)
    x=np.matrix(x)
    return x
#signal normalization
def normalized(x):
    norm_a=[]
    sum_x=np.sum(x)
    for i in x:
        norm_a.append(float(i)/sum_x)
    return norm_a

#generate the sampling score based on graph structure
def sampling_score(N,Fourier_basis_K,sampling_mode):
    if sampling_mode=='uniform_sampling':
        dist=float(1)/N
        norm_dist=dist*np.ones(N)
    if sampling_mode=='leverage_score':
        dist=[]
        for i in Fourier_basis_K:
            dist.append(np.sqrt(np.linalg.norm(i)))
        norm_dist=normalized(dist)
        norm_dist=np.asarray(norm_dist)
    if sampling_mode=='square_root_leverage_score':
        dist=[]
        for i in Fourier_basis_K:
            dist.append(np.linalg.norm(i))
        norm_dist=normalized(dist)
        norm_dist=np.asarray(norm_dist)
    return norm_dist

#sacle_matrix generation based on sampling score
def sacle_matrix(sample_score,m):
    Scaling_matrix=[]
    for i in sample_score:
        a=(float(1)/np.sqrt(m*i))
        Scaling_matrix.append(a)
    Scaling_matrix=np.asarray(Scaling_matrix) 
    Scaling_matrix=np.diag(Scaling_matrix)
    return Scaling_matrix

def sample_operator(x,m,sample_score):

    N=len(x)
    sampling_operator=np.zeros((m,N))
    sample_index=np.random.choice(N,m, replace=False, p=sample_score)
    for i in range(m):
        sampling_operator[i][sample_index[i]]=1
    sampling_operator=np.matrix(sampling_operator)
    return sampling_operator

def signal_recovery(signal_g,Fourier_basis,K,epsilon,sampling_mode):
    Fourier_basis_K=Fourier_basis[:,0:K]
    Inv_fourier_basis=np.linalg.inv(Fourier_basis)
    Inv_fourier_basis_K=Inv_fourier_basis[0:K,:]
    sample_score=sampling_score(N,Fourier_basis_K,sampling_mode)
    Scaling_matrix=sacle_matrix(sample_score,m)
    sampling_operator=sample_operator(signal_g,m,sample_score)
    x1=signal_g.reshape(N,1)
    x1=np.matrix(x1)
    noise=np.random.normal(0,epsilon)
    noise=np.ones((m,1))*noise
    y=sampling_operator*x1+noise
    x_recovery=Fourier_basis_K*Inv_fourier_basis_K*np.transpose(sampling_operator)\
    *sampling_operator*Scaling_matrix*Scaling_matrix\
    *np.transpose(sampling_operator)*y
    
    return x_recovery

def comparison_plot(N):
    uniform_sampling_mse=pd.read_csv('MSE uniform_sampling.csv')
    squre_root_leverage_mse=pd.read_csv('MSE square_root_leverage_score.csv')
    leverage_score_mse=pd.read_csv('MSE leverage_score.csv')
    
    plt.plot(m_value,np.log(uniform_sampling_mse['MSE']),label='uniform sampling')
    plt.plot(m_value,np.log(squre_root_leverage_mse['MSE']),label='square root leverage score')
    plt.plot(m_value,np.log(leverage_score_mse['MSE']),label='leveragae score')
    plt.legend(loc='upper right', shadow=True)
    plt.title('log mean square error from different sampling method')
    plt.xlabel('sample number')
    plt.ylabel('log MSE')

#=================================Main function======================================================
#generate graph fourier transform basis

N=5000 #node number in the network
K=10 #band limit
beta=0.5 #spectral decay factor
epsilon=10**-4  #noise introducing into the system N~(0,epsilon)
base_mode='adjacency'
graph_type='ring graph'#'random geometric graph'    #'ring graph'
sampling_mode='uniform_sampling' #uniform_sampling or square_root_leverage_score

Inv_fourier_basis=graph_generation(N,graph_type,base_mode)
Fourier_basis=np.linalg.inv(Inv_fourier_basis)
signal_g=graph_signal_generation(N,K,beta,Inv_fourier_basis)

sampling_mode='square_root_leverage_score' 
MSE=[]
Recovery_signal=[]

m_value=range(100,N,300)
for m in m_value:
    print 'm=',m
    K_recovery=max(int(m**(float(1)/(2*beta+1))),10) #recovery bandwidth
    print K_recovery
    x_recovery=signal_recovery(signal_g,Inv_fourier_basis,K_recovery,epsilon,sampling_mode)
    norm=np.linalg.norm(x_recovery-signal_g)
    print norm
    MSE.append(norm)
    Recovery_signal.append(x_recovery)
#==================================Save file==============================================================   
MSE1=pd.Series(MSE)
file_name='MSE '+sampling_mode+'.csv'  
MSE1.to_csv(file_name,index=False,header=['MSE'])
    
#==================================Plot figures==============================================================  
fig1=plt.figure(figsize=(7,5))
plt.plot(signal_g,label='x original')
plt.plot(x_recovery,label='x recovery')
legend = plt.legend(loc='upper right', shadow=True)
plt.title('comparison between recovery signal and original signal')

fig2=plt.figure(figsize=(7,5))
plt.plot(m_value,np.log(MSE))
title='Log mean square error for '+sampling_mode
plt.title(title)
plt.xlabel('sample number')
plt.ylabel('log MSE')

#comparison_plot(N)





