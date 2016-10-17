# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 23:58:35 2016

@author: Subhajit
"""


import numpy as np
import scipy.io
import h5py

def load_matfile(filename='./data/indian_pines_data.mat'):
    f = h5py.File(filename)
    #print f['X_r'].shape
    if 'pca' in filename:
        X=np.asarray(f['X_r'],dtype='float32')
    else:
        X=np.asarray(f['X'],dtype='float32')
    y=np.asarray(f['labels'],dtype='uint8')
    f.close()
    return X,y
    


if __name__=='__main__':
    X,y=load_matfile(filename='./data/indian_pines_data_pca.mat')
    