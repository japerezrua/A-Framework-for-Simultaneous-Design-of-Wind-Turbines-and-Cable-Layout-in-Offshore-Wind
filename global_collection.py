# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:51:10 2020

@author: juru
"""
import numpy as np
import cplex
from var import var
from collection_system_global import collection_system
from arranging import arranging
from stopping import stopping
from warm_starting import warm_starting 

def global_optimizer(WTc,X,Y,Cables,C=200,OSSc=1,T=[5,15,20,25,30,35,40,45,50],gap=1,iterative=True,time_limit=3600):
    #%% Pre-processing
    UL=max(Cables[:,1])
    var_output_general=[]
    solution_output_general=[]
    substation_number_general=[]
    warm=cplex.SparsePair([],[])
    var_output,objective,cable_selected=var(WTc,OSSc,X,Y,UL,Cables,T[0],1)
    gap_outputit,time_formulating,time_solving,solutions=[],[],[],[]
    for i in range(len(T)):
        print("Iteration number global solver:",i)
    #%% Solving the model
        solution_var,solution_value,elapsed_form,elapsed_solved,gap_output=collection_system(i,WTc,OSSc,X,Y,UL,Cables,C,
        var_output,objective,cable_selected,gap,warm,time_limit)
        solutions+=[solution_value/1000000]
        gap_outputit+=[100*gap_output]
        time_formulating+=[elapsed_form/60]
        time_solving+=[elapsed_solved/60]
    #%% Arranging results
        variables_results=np.array(solution_var)
        b,substation_number,var_output_general,solution_output_general,substation_number_general=arranging(OSSc,
        variables_results,var_output,cable_selected,objective,var_output_general,solution_output_general,
        substation_number_general)   
    #%% Stopping criterion
        if stopping(i,b,var_output_general,iterative):
           #iteration=i
           print("Final iteration number global solver:",i)
           break
    #%% Warm-starting for next iteration
        if i<len(T)-1:
          var_output,objective,cable_selected=var(WTc,OSSc,X,Y,UL,Cables,T[i+1],0)  
          warm=warm_starting(i+1,solution_output_general,var_output,substation_number_general)
        if i==len(T)-1:
           print("Final iteration number global solver:",i) 
    return b,solution_value,gap_outputit,time_formulating,time_solving,solutions