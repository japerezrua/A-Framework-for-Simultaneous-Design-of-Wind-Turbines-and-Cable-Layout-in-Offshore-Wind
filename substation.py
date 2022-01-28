# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:15:22 2020

@author: juru
"""

import numpy as np

def substation_location(n_wt,X,Y,Diameter,Correction):
    oss=np.zeros(2)
    if Correction:
        oss_x,oss_y=sum(X)/n_wt,sum(Y)/n_wt
        dist_to_oss=np.sqrt((X-oss_x)**2+(Y-oss_y)**2)
        args=np.argsort(dist_to_oss[:])
        dist_to_oss=dist_to_oss[args]
        X_sorted = X[args]
        Y_sorted = Y[args]
        if dist_to_oss[0]<1.4142*Diameter:
           oss[0],oss[1]=sum(X_sorted[0:4])/4,sum(Y_sorted[0:4])/4
        else:
            oss[0],oss[1]=oss_x,oss_y
           #print(dist_to_oss[0])
        #dist_to_oss_corrected=np.zeros((len(X),1))
        #for i in range(len(X)):
        #    dist_to_oss_corrected[i]=np.sqrt((X[i]-oss[0])**2+(Y[i]-oss[1])**2)
        #args2=np.argsort(dist_to_oss_corrected[:,0])
        #dist_to_oss_corrected=dist_to_oss_corrected[args2]
        #X_sorted_corrected = X[args2]
        #Y_sorted_corrected = Y[args2]        
    else:
        oss[0],oss[1]=sum(X)/n_wt,sum(Y)/n_wt
    return oss
        
    
    
    