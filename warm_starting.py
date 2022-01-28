# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:49:20 2020

@author: juru
"""

import numpy as np
import cplex

def warm_starting(i,solution_output_general,var_output,substation_number_general):
    b=solution_output_general[i-1]
    pos1=list(range(substation_number_general[i-1].shape[0]))
    val1=[x for x in substation_number_general[i-1]]
    pos2,pos3,val2,val3=[],[],[],[]
    for j in range(b.shape[0]):
        pos2_aux=list(np.where((var_output[:,0]==b[j,0]) & (var_output[:,1]==b[j,1]) & (var_output[:,3]==0))[0])
        pos2_aux=[x.item() for x in pos2_aux]
        pos3_aux=list(np.where((var_output[:,0]==b[j,0]) & (var_output[:,1]==b[j,1]) & (var_output[:,3]==b[j,4]))[0])
        pos3_aux=[x.item() for x in pos3_aux]
        val2_aux=[1]*len(pos2_aux)
        val3_aux=[1]*len(pos3_aux)
        pos2=pos2+pos2_aux
        pos3=pos3+pos3_aux
        val2=val2+val2_aux
        val3=val3+val3_aux    
    warm=cplex.SparsePair(pos1+pos2+pos3,val1+val2+val3)
    #warm=cplex.SparsePair(pos1,val1)
    return warm        
        