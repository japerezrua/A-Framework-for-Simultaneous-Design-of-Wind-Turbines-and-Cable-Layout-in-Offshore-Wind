# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:50:56 2020

@author: juru
"""
import numpy as np
import matplotlib.pyplot as plt

def plotting(X,Y,Cables,b):
    plt.figure()
    plt.plot(X[1:], Y[1:], 'r+',markersize=10, label='Turbines')
    plt.plot(X[0], Y[0], 'ro',markersize=10, label='OSS')
    X,Y=np.array(X),np.array(Y)
    for i in range(len(X)):
        plt.text(X[i]+50, Y[i]+50,str(i+1))
    colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
    for i in range(Cables.shape[0]):
        index = b[:,3]==i
        if index.any():
           n1xs = X[b[index,0].astype(int)-1]
           n2xs = X[b[index,1].astype(int)-1]
           n1ys = Y[b[index,0].astype(int)-1]
           n2ys = Y[b[index,1].astype(int)-1]
           xs = np.vstack([n1xs,n2xs])
           ys = np.vstack([n1ys,n2ys])
           plt.plot(xs,ys,'{}'.format(colors[i]))
           plt.plot([],[],'{}'.format(colors[i]),label='Cable: {} mm2'.format(Cables[i,0]))
    plt.legend()
    return 