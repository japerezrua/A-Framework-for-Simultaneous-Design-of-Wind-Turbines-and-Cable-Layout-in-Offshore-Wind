# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 08:07:37 2020

@author: juru
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from c_mst import capacitated_spanning_tree
import time

def cmst_cables(X=[],Y=[],T=[],Cables=[],plot=True):
    """
    Assigns cables to the obtained C-MST in previous stage

    Parameters
    ----------
    *X, Y: Arrays([n_wt_oss]) 
            X,Y positions of the wind turbines and oss
    *T: Obtained tree in previous stage
    *Cables: Array([cables available, 3]). Each row is a cable available. Column number one is cross-section, column number 2 number of WTs
                                           and column number 3 is the price/km of the cable
    
    :return: Td: Array. Each row is a connection. First column is node 1, second column node 2, third column length (m), fourd column is
                        cable type (index of the Cables array), and last column is the cost of the edge

    """   
    G = nx.Graph() 
    G.add_nodes_from([x+1 for x in range(len(T[:,1])+1)])
    G.add_edges_from([tuple(T[edge,0:2]) for edge in range(len(T[:,1]))]) 
    T_d=np.array([x for x in nx.dfs_edges(G, source=1)]).astype(int)
    accumulator=np.zeros(T_d.shape[0])
    for j in range(len(T[:,1])):
        k = j + 2
        continue_ite = 1
        look_up = k
        while continue_ite:
              accumulator += (T_d[:,1]==look_up).astype(int)
              if (T_d[:,1]==look_up).astype(int).sum() > 1:
                 Exception('Error')
              if T_d[(T_d[:,1]==look_up)][0,0] == 1:
                 continue_ite = 0
              else:
                  look_up=T_d[(T_d[:,1]==look_up)][0,0]
    T_d=np.append(T_d,np.zeros((accumulator.shape[0],3)),axis=1)
    for k in range(T_d.shape[0]):
        aux1=np.argwhere((T[:,0]==T_d[k,0]) & (T[:,1]==T_d[k,1]))
        aux2=np.argwhere((T[:,1]==T_d[k,0]) & (T[:,0]==T_d[k,1]))
        if aux2.size==0:
           T_d[k,2]=T[aux1,2]
        if aux1.size==0:
           T_d[k,2]=T[aux2,2]
        if (aux2.size==0) and (aux1.size==0):
            Exception('Error')
        for k in range(accumulator.shape[0]):
            for l in range(Cables.shape[0]):
                if accumulator[k]<=Cables[l,1]:
                   break  
            T_d[k,3]=l
        for k in range(T_d.shape[0]):
            T_d[k,4]=(T_d[k,2]/1000)*Cables[T_d.astype(int)[k,3],2]
    if plot:
        plt.figure()
        plt.plot(X[1:], Y[1:], 'r+',markersize=10, label='Turbines')
        plt.plot(X[0], Y[0], 'ro',markersize=10, label='OSS')
        for i in range(len(X)):
            plt.text(X[i]+50, Y[i]+50,str(i+1))
        colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
        for i in range(Cables.shape[0]):
            index = T_d[:,3]==i
            if index.any():
               n1xs = X[T_d[index,0].astype(int)-1].ravel().T
               n2xs = X[T_d[index,1].astype(int)-1].ravel().T
               n1ys = Y[T_d[index,0].astype(int)-1].ravel().T
               n2ys = Y[T_d[index,1].astype(int)-1].ravel().T
               xs = np.vstack([n1xs,n2xs])
               ys = np.vstack([n1ys,n2ys])
               plt.plot(xs,ys,'{}'.format(colors[i]))
               plt.plot([],[],'{}'.format(colors[i]),label='Cable: {} mm2'.format(Cables[i,0]))
        plt.legend()
    return T_d
if __name__ == "__main__":
  t = time.time()
  X=[387100,383400,383400,383900,383200,383200,383200,383200,383200,383200,383200,383200,383300,384200,384200,384100,384000,383800,383700,383600,383500,383400,383600,384600,385400,386000,386100,386200,386300,386500,386600,386700,386800,386900,387000,387100,387200,383900,387400,387500,387600,387800,387900,388000,387600,386800,385900,385000,384100,384500,384800,385000,385100,385200,385400,385500,385700,385800,385900,385900,385500,385500,386000,386200,386200,384500,386200,386700,386700,386700,384300,384400,384500,384600,384300,384700,384700,384700,385500,384300,384300]
  Y=[6109500,6103800,6104700,6105500,6106700,6107800,6108600,6109500,6110500,6111500,6112400,6113400,6114000,6114200,6115100,6115900,6116700,6118400,6119200,6120000,6120800,6121800,6122400,6122000,6121700,6121000,6120000,6119100,6118100,6117200,6116200,6115300,6114300,6113400,6112400,6111500,6110700,6117600,6108900,6108100,6107400,6106300,6105200,6104400,6103600,6103600,6103500,6103400,6103400,6104400,6120400,6119500,6118400,6117400,6116500,6115500,6114600,6113500,6112500,6111500,6105400,6104200,6110400,6109400,6108400,6121300,6107500,6106400,6105300,6104400,6113300,6112500,6111600,6110800,6110100,6109200,6108400,6107600,6106500,6106600,6105000]  
  X=np.array(X)
  Y=np.array(Y)
  
  option=3
  UL=11
  Inters_const=False
  Cables=np.array([[500,3,100000],[800,5,150000],[1000,10,250000]])
  
  T,feasible=capacitated_spanning_tree(X,Y,option,UL,Inters_const)

  print("The total length of the solution is {value:.2f} m".format(value = sum(T[:,2])))
  print("Feasibility: {feasible1}".format(feasible1=feasible)) 
  
  if feasible:
      T_cables=cmst_cables(X,Y,T,Cables)
      print("The total cost of the system is {value:.2f} Euros".format(value = T_cables[:,-1].sum()))
  elapsed = time.time() - t
  print("The total time is {timep: .2f} s".format(timep=elapsed))
  

  
  
