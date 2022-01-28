# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:27:32 2020

@author: juru
"""
import numpy as np

def stopping(i,b,var_output_general,iterative):    
    if i>1:
        var_output=var_output_general[i-1]
        condition=1
        for j in range(b.shape[0]):
            pos1=list(np.where((var_output[:,0]==b[j,0]) & (var_output[:,1]==b[j,1]) & (var_output[:,3]==b[j,4]))[0])
            if len(pos1)==0:
               condition=0
               break
    elif (i==1) and (not iterative):
        condition=1
    else:
        condition=0
    return condition