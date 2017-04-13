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
#graph generation
def graph_generation(N,graph_type='ring graph',base_mode='adjacency'):
    if graph_type=='ring graph':
        k=4   #neighbor number
        p=0
        G=nx.newman_watts_strogatz_graph(N, k, p, seed=None)    #ring graph
        fourier_basis=normalized_fourier_basis(G,base_mode='adjacency')
    if graph_type=='random graph':
        G = nx.erdos_renyi_graph(N,0.1)                         #random graph(Erdos-Renyi Graph)
        fourier_basis=normalized_fourier_basis(G,base_mode='adjacency')
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
def normalized_fourier_basis(G,base_mode='adjacency'):
    if base_mode=='adjacency':
        A=nx.adjacency_matrix(G, nodelist=None, weight='weight').todense() 
        Sigma_A, V_A = np.linalg.eigh(A) 
        index_sig = np.argsort(Sigma_A)
        Sigma_A = Sigma_A[index_sig[::-1]] #sorting the eigenvalue
        V_A = V_A[:,index_sig[::-1]] #sorting the according eigenvectors
        #Normalized matrix A
        A=A/Sigma_A[0] #normalized matrix A
        Sigma_A, V_A = np.linalg.eigh(A) 
        index_sig = np.argsort(Sigma_A)
        Sigma_A = Sigma_A[index_sig[::-1]]
        Fourier_basis = V_A[:,index_sig[::-1]]
    if base_mode=='Laplacian':
        Laplacian = nx.normalized_laplacian_matrix(G).todense()
        Sigma_L, V_L = np.linalg.eigh(Laplacian)
        index_sig = np.argsort(Sigma_L)
        Sigma_L = Sigma_L[index_sig[::-1]]
        V_L = V_L[:,index_sig[::-1]]
        #Normalized Laplacian matrix
        Laplacian=Laplacian/Sigma_L[0] #normalized matrix A
        Sigma_L, V_L = np.linalg.eigh(Laplacian) 
        index_sig = np.argsort(Sigma_L)
        Sigma_L = Sigma_L[index_sig[::-1]]
        Fourier_basis = V_L[:,index_sig[::-1]]
    return Fourier_basis
    
#signal generation
def graph_signal_generation(n,K,beta,Fourier_basis):
    N=n
    K=10
    signal=[]
    for i in range(N):
        if i<K:
            signal.append(np.random.normal(1,0.5*0.5))
        else:
            signal.append((float(K)/i)**(2*beta))
            #signal.append(0)  #test the completely band limited case
    signal=np.asarray(signal)
    signal_norm=np.linalg.norm(signal)  #fourier transform x_f=U_A*x
    signal_norm=signal/signal_norm
    signal_norm=np.transpose(signal_norm)
    signal_norm=signal_norm.reshape(len(signal),1)
    x=Fourier_basis*signal_norm
    x=np.asarray(x)
    x=x.reshape(n,)
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
    if sampling_mode=='uniform sampling':
        dist=float(1)/N
        norm_dist=dist*np.ones(N)
    if sampling_mode=='experiment design':
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
    #Scaling_matrix=(float(1)/np.sqrt(m*distribution))*np.ones((N,)) #unidorm sampling
    Scaling_matrix=np.diag(Scaling_matrix)
    return Scaling_matrix

def sample_operator(x,m):

    N=len(x)
    sampling_operator=np.zeros((m,N))
    sample_index=np.random.choice(N,m, replace=False, p=None)
    for i in range(m):
        sampling_operator[i][sample_index[i]]=1
    sampling_operator=np.matrix(sampling_operator)
    return sampling_operator
#================================================================================================
#generate graph fourier transform basis
N=1000
K=10 #band limit
beta=0.5 #control the frequency band outside the bandlimit
Fourier_basis=graph_generation(N,graph_type='ring graph',base_mode='adjacency')
Inv_fourier_basis=np.linalg.inv(Fourier_basis)
signal_g=graph_signal_generation(N,K,beta,Fourier_basis)

MSE=[]
Recovery_signal=[]

m_value=range(100,N,100)
for m in m_value:
    print 'm=',m
    K=max(int(m**(float(1)/(2*beta+1))),10)
    K=30
    Fourier_basis_K=Fourier_basis[:,0:K]
    Inv_fourier_basis_K=Inv_fourier_basis[0:K,:]
    sample_score=sampling_score(N,Fourier_basis_K,sampling_mode='uniform sampling')
    Scaling_matrix=sacle_matrix(sample_score,m)
    
    sampling_operator=sample_operator(signal_g,m)
    x1=signal_g.reshape(N,1)
    x1=np.matrix(x1)
    SNR=np.random.normal(0,10**-4)
    noise=np.ones((m,1))*SNR
    y=sampling_operator*x1+noise
    x_recovery=Fourier_basis_K*Inv_fourier_basis_K*np.transpose(sampling_operator)\
    *sampling_operator*Scaling_matrix*Scaling_matrix\
    *np.transpose(sampling_operator)*y
    norm=np.linalg.norm(x_recovery-x1)
    print norm
    MSE.append(norm)
    Recovery_signal.append(x_recovery)
    
fig1=plt.figure(1)
plt.plot(signal_g,label='x original')
plt.plot(x_recovery,label='x recovery')
legend = plt.legend(loc='upper right', shadow=True)
plt.title('comparison between recovery signal and original signal')

fig2=plt.figure(2)
plt.plot(m_value,np.log(MSE))
plt.title('Log mean square error')
plt.xlabel('sample number')
plt.ylabel('log MSE')

MSE1=pd.Series(MSE)
file_name='MSE_N_'+str(N)+'_beta_'+str(beta)
MSE1.to_csv('MSE.csv')

    
#generate graph signal
