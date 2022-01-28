# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:54:03 2020

@author: juru
"""

import numpy as np

def cost_function(wt_power,voltage,turb_num):
    if voltage==33:
       Ap,Bp,Cp=0.411e6,0.596e6,4.1
    elif voltage==66:
       Ap,Bp,Cp=0.688e6,0.625e6,2.05
    else:
        raise Exception('Supported rated voltages are 33 kV or 66 kV. The value of voltage was:: {}'.format(voltage))    
    cable_power=wt_power*turb_num*1000
    cost=1.26*(((Ap)+((Bp)*(np.exp((Cp*cable_power)/(1e8)))))/(9.0940)) #Cost in Euro/km
    return cost

def cost_function_array(wt_power,voltage,Cables):
    Cables=np.concatenate((Cables,np.full((Cables.shape[0],1),0)), axis=1)
    for i in range(Cables.shape[0]):
        Cables[i,2]=cost_function(wt_power,voltage,Cables[i,1])
    return Cables