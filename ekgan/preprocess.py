import numpy as np
import os
import pandas as pd
import numpy as np

from scipy import signal
import librosa
from numpy.random import random
from scipy.linalg import sqrtm
from sklearn.metrics import mean_squared_error
from math import sqrt


time_len = 512

def load_data(sample_x, sample_y):
    '''
    '''
    X = []
    Y = []
    list_dir=[]
    time_len = 512
    
    for idx, i in enumerate(sample_x):
        if idx % 10 == 0:
            print(idx, '/', len(sample_x))
        try:
            df = pd.read_csv(f'/home/ghjoo/data/lbrbafmi/lbrbafmi/data_lbrbaf/dis/' + i)
        except:
            df = pd.read_csv(f'/home/ghjoo/data/lbrbafmi/lbrbafmi/data_lbrbaf/nsr/' + i)
        df = np.transpose(np.array(df),(1,0))
        
        df_12 = np.vstack((df[0],df[0],df[0],df[0],df[0],df[0],df[0],df[0],df[0],df[0],df[0],df[0]))
        #print(df_12.shape)
        
        X.append(df_12)
        Y.append(np.array(df))
        list_dir.append(i)
    X = np.array(X)
    Y = np.array(Y)
    list_dir = np.array(list_dir)
    X = np.expand_dims(X,axis=-1)
    Y = np.expand_dims(Y,axis=-1) 
    X = np.pad(X,[(0,0),(2,2),(0,0),(0,0)], mode='constant', constant_values=(0))
    Y=np.pad(Y,[(0,0),(2,2),(0,0),(0,0)], mode='constant', constant_values=(0))

    return X,Y
