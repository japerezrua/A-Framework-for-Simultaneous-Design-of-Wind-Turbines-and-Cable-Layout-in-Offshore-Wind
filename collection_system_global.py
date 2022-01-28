# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 12:57:20 2020

@author: juru
"""
import cplex
import numpy as np
import math
from var import var
from two_lines_intersecting import two_lines_intersecting
import time

def collection_system(big_ite,WTc,OSSc,X,Y,UL,Cables,C,var_output,objective,cable_selected,gap,warm,time_limit):
    #%% Model
    t = time.time()
    model = cplex.Cplex()
    model.parameters.mip.tolerances.mipgap.set(gap/100) 
    model.parameters.timelimit.set(time_limit)
    model.parameters.mip.limits.treememory.set(6500)
    #%% Variables
    model.variables.add(types='I'*OSSc+'B'*(var_output.shape[0]-OSSc))
    model.variables.set_upper_bounds([(i,math.ceil(WTc/OSSc)) for i in range(OSSc)])
    #%% Objective function
    model.objective.set_sense(model.objective.sense.minimize)
    model.objective.set_linear([(i,float(value)) for i,value in enumerate(objective)])
    #%% Constraints
    #%% C1: OSSs supporting all WTs
    lhsC1 = cplex.SparsePair(list(range(OSSc)),[1]*OSSc)
    model.linear_constraints.add(lin_expr=[lhsC1], senses=["E"], rhs=[WTc])
    #%% C2: OSSs clustering WTs
    lhsC2=[]
    for i in range(OSSc):
        initial=i*WTc*(UL+1)+OSSc
        final=(i+1)*WTc*(UL+1)+OSSc
        pos_aux=list(range(initial,final))
        val_aux=list(var_output[pos_aux,3])
        pos=[i]+pos_aux
        val=[-1]+val_aux
        lhsC2.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC2, senses=["E"]*len(lhsC2), rhs=[0]*len(lhsC2))
    #%% C3: Limiting number of main feeders to the OSSs
    lhsC3=[]
    for i in range(OSSc):
        initial=i*WTc*(UL+1)+OSSc
        final=(i+1)*WTc*(UL+1)+OSSc
        pos=list(range(initial,final))
        val=[1]*len(pos)
        for j in range(0,len(val_aux),UL+1):
            val[j]=0
        lhsC3.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC3, senses=["L"]*len(lhsC3), rhs=[C]*len(lhsC3))
    #%% C4: Tree topology
    lhsC4=[]
    for i in range(OSSc+1,WTc+OSSc+1):       
        pos=list(np.where((var_output[:,1]==i) & (var_output[:,3]>0))[0])
        pos=[x.item() for x in pos]
        val=[1]*len(pos)
        lhsC4.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC4, senses=["E"]*len(lhsC4), rhs=[1]*len(lhsC4))
    #%% C5: Flow conservation
    lhsC5=[]
    for i in range(OSSc+1,WTc+OSSc+1):
        pos1=list(np.where((var_output[:,1]==i) & (var_output[:,3]>0))[0])
        pos1=[x.item() for x in pos1]
        pos2=list(np.where((var_output[:,0]==i) & (var_output[:,3]>0))[0])
        pos2=[x.item() for x in pos2]
        val1=list(var_output[pos1,3])
        val2=list(-1*var_output[pos2,3])           
        pos=pos1+pos2
        val=val1+val2
        lhsC5.append(cplex.SparsePair(pos,val))
        model.linear_constraints.add(lin_expr=lhsC5, senses=["E"]*len(lhsC5), rhs=[1]*len(lhsC5))
    #%% C6: Deactivating crossing arcs
    lhsC6=[]
    for i in range(1,WTc+OSSc+1):
        for j in range(1,WTc+OSSc+1):
            pos1=list(np.where((var_output[:,0]==i) & (var_output[:,1]==j) & (var_output[:,3]>0))[0])
            pos1=[x.item() for x in pos1]
            pos2=list(np.where((var_output[:,0]==i) & (var_output[:,1]==j) & (var_output[:,3]==0))[0])
            pos2=[x.item() for x in pos2]
            if len(pos2)>1:
                raise Exception("Not possible")   
            val1=[1]*len(pos1)
            val2=[-1]*len(pos2)
            pos=pos1+pos2
            val=val1+val2            
            lhsC6.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC6, senses=["L"]*len(lhsC6), rhs=[0]*len(lhsC6)) 
    #%% C7: Defining crossing arcs
    size_var=var_output.shape[0]
    lhsC7=[]
    for i in range(size_var):
        if var_output[i,0]!=0 and var_output[i,1]!=0 and var_output[i,3]==0:
           for j in range(i+1,size_var):
               if var_output[j,3]==0 and i!=j:
                  line1=np.array([[X[var_output[i,0].astype(int)-1],Y[var_output[i,0].astype(int)-1]],\
                            [X[var_output[i,1].astype(int)-1],Y[var_output[i,1].astype(int)-1]]])
                  line2=np.array([[X[var_output[j,0].astype(int)-1],Y[var_output[j,0].astype(int)-1]],\
                                [X[var_output[j,1].astype(int)-1],Y[var_output[j,1].astype(int)-1]]])
                  if two_lines_intersecting(line1,line2):
                     pos1=[i,j]
                     val1=[1]*len(pos1)
                     potential1=list(np.where((var_output[i,0]==var_output[:,1]) & (var_output[i,1]==var_output[:,0]) & (0==var_output[:,3]))[0])
                     pos2=[x.item() for x in potential1]
                     val2=[1]*len(pos2)
                     potential2=list(np.where((var_output[j,0]==var_output[:,1]) & (var_output[j,1]==var_output[:,0]) & (0==var_output[:,3]))[0])
                     pos3=[x.item() for x in potential2]
                     val3=[1]*len(pos3)
                     pos=pos1+pos2+pos3
                     val=val1+val2+val3
                     lhsC7.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC7, senses=["L"]*len(lhsC7), rhs=[1]*len(lhsC7))       
    #%% C8: Valid inequalities 
    lhsC8=[]    
    for i in range(OSSc+1,WTc+OSSc+1):
        for j in range(2,UL):
            pos1=list(np.where((var_output[:,1]==i) & (var_output[:,3]>=j+1) & (var_output[:,3]<=UL))[0])
            pos1=[x.item() for x in pos1]
            pos2=list(np.where((var_output[:,0]==i) & (var_output[:,3]>=j) & (var_output[:,3]<=UL-1))[0])
            pos2=[x.item() for x in pos2] 
            val1=[1]*len(pos1)
            val2=[1]*len(pos2)
            for k in range(len(pos1)):
                val1[k]=-math.floor((var_output[pos1[k],3]-1)/j)
            pos=pos1+pos2
            val=val1+val2            
            lhsC8.append(cplex.SparsePair(pos,val))
    model.linear_constraints.add(lin_expr=lhsC8, senses=["L"]*len(lhsC8), rhs=[0]*len(lhsC8))   
    #%% Running the model
    if big_ite>0:
       model.MIP_starts.add(warm,model.MIP_starts.effort_level.repair,'previous_iteration')
    elapsed_form = time.time() - t
    t = time.time()
    model.solve()
    elapsed_solved = time.time() - t
    solution_var=model.solution.get_values()
    solution_value=model.solution.get_objective_value()
    gap_output=model.solution.MIP.get_mip_relative_gap()
    del model
    try:
        from IPython import get_ipython
        get_ipython().magic('clear')
    except:
        pass
    return solution_var,solution_value,elapsed_form,elapsed_solved,gap_output
if __name__ == "__main__":
    WTc=10
    OSSc=1
    X=[10000,8285.86867841725,9097.62726321941,8736.91009361509,9548.66867841725,10360.4272632194,9187.95150881294,9999.71009361510,10811.4686784173,10450.7515088129,11262.5100936151]
    Y=[10000,9426,8278,10574,9426,8278,11722,10574,9426,11722,10574]  
    UL=6
    Cables=np.array([[500,2,100000],[800,4,150000],[1000,6,250000]])
    C=5
    T=5
    feasibility=0
    var_output,objective,cable_selected=var(WTc,OSSc,X,Y,UL,Cables,T,feasibility)
    solution_var,solution_value,elapsed,gap_output=collection_system(WTc,OSSc,X,Y,UL,Cables,C,T,var_output,objective,cable_selected)