#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 20:48:56 2017

@author: YingxueZhang
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
MSE_10000_beta_05=pd.read_csv('MES_10000_beta_05_new.csv',header=None)
MSE_10000_beta_1=pd.read_csv('MES_10000_beta_1_new.csv',header=None)
log_mse_05=np.log(MSE_10000_beta_05[1])
log_mse_1=np.log(MSE_10000_beta_1[1])

mse_05=MSE_10000_beta_05[1]
mse_1=MSE_10000_beta_1[1]

m_value=range(990,10000,1000)
fig1=plt.figure(1)
#plt.plot(m_value,mse_05[0:-1:10],label='beta=0.5')
#plt.plot(m_value,mse_05,label='beta=0.5')
plt.plot(m_value,mse_05,label='beta=0.5')
#plt.title('mean square error comparison')
plt.title('mean square error when beta=1 N=10000')
legend = plt.legend(loc='upper right', shadow=True)
plt.xlabel('sample number')
plt.ylabel('MSE')


