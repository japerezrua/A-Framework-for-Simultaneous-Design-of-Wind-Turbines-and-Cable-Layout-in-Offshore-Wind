# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 17:54:14 2020

@author: juru
"""
import numpy as np

def arranging(OSSc,variables_results,var_output,cable_selected,objective,var_output_general,solution_output_general,substation_number_general):   
    b=np.zeros((0,6))
    for j in range(OSSc+1,len(variables_results)):
        if variables_results[j]<=1.01 and variables_results[j]>=0.99 and var_output[j,3]!=0:
           b_aux=np.zeros((1,6))
           b_aux[0,0],b_aux[0,1],b_aux[0,2]=var_output[j,0],var_output[j,1],var_output[j,2]
           b_aux[0,3],b_aux[0,4],b_aux[0,5]=cable_selected[j],var_output[j,3],objective[j]
           b=np.vstack((b,b_aux))
    substation_number=variables_results[:OSSc]
    var_output_general.append(var_output)
    solution_output_general.append(b)
    substation_number_general.append(substation_number)
    return b,substation_number,var_output_general,solution_output_general,substation_number_general