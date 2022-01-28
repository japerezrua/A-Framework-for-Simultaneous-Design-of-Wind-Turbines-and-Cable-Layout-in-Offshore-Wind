# -*- coding: utf-8 -*-
"""
Created on Thu May 28 08:12:54 2020

@author: juru
"""
import numpy as np
from intersection_checker import intersection_checker
import matplotlib.pyplot as plt

def capacitated_spanning_tree(X=[],Y=[],option=3,UL=100,Inters_const=True,max_it=20000):
    """
    Calculate a minimum spanning tree distance for a layout.
    Capacitated minimum spanning tree heuristics algorithm for Topfarm.

    Parameters
    ----------
    *X, Y: list([n_wt_oss]) or array type as well
            X,Y positions of the wind turbines and oss
    *option: Heuristic type. option=1 is Prim. option=2 is Kruskal. option=3 is Esau-Williams
    *max_it: Maximm number of iterations for the heuristics
    *UL: Upper limit for the max number of wind turbines connectable by the biggest available cable
    *Inters_const=Bool. True is cable crossings are not allowed. False if they are allowed.
    
    :return: T: Array. First column is first node, second column is second noce, third column is the distance between nodes
                       The OSS is always the node number 1. The WTs go from 2 to number of WTs plus one
             feasible: Bool. True is solution is feasible. False if not.

    """  
#%%  Initializing arrays, lists, variables (until line 46 .m file)  
    n_wt_oss=len(X)  #Defining number of wind turbines with OSS
    half=int(n_wt_oss*(n_wt_oss-1)/2)  
    edges_tot=np.zeros((2*half,5)) #Defining the matrix with Edges information
    cont_edges=0
    for i in range(n_wt_oss):
        for j in range(i+1,n_wt_oss):
            edges_tot[cont_edges,0]=i+1 #First element is first node (Element =1 is the OSS. and from 2 to Nwt the WTs)
            edges_tot[cont_edges,1]=j+1 #Second element is second node
            edges_tot[cont_edges,2]=np.sqrt((X[j]-X[i])**2+(Y[j]-Y[i])**2) #Third element is the length of the edge
            cont_edges+=1
    CP=[x for x in range(n_wt_oss)] #Initializing component position list for each node. A component goes from 0 until n_wt_oss-1. Fixed length.
    address_nodes=[-1 for x in range(n_wt_oss)] #Initializing address list for each node. It indicates the root node for each node in the tree and in subtrees from OSS. Fixed length.
    address_nodes[0]=0
    address_nodes=np.array(address_nodes,dtype=int)
    C=[[x+1] for x in range(n_wt_oss)] #Initializing component list (nodes belonging to each comonent). A component goes from 0 until n_wt_oss-1, and its length decreases until 1 (component 0). Variable length.
    S=[1 for x in range(n_wt_oss)] #Initializing size of components list (how many nodes are in each component). A component goes from 0 until n_wt_oss-1, and its length decreases until 1 (component 0 with n_wt_oss-1 elements). Variable length.
    go,it,node1,node2,weight = True,0,0,0,np.zeros((n_wt_oss,1)) #Initializing variables for iterations 
    if option == 1: #Initializing weight of nodes. Each index represents a node, such as Node=Index+1
       weight[0],weight[1:n_wt_oss]=0,-10**50
    elif option == 2:
       weight
    elif option == 3:
       weight[0],weight[1:n_wt_oss]=0,edges_tot[0:n_wt_oss-1,2].reshape(n_wt_oss-1,1)           
    else:
        raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option))
    for i in range(2*half):#Forming the big matrix with all edges and corresponding trade-off values (fixed size).
        if i<=half-1:
           edges_tot[i,3]=weight[edges_tot[i,0].astype(int)-1,0]
           edges_tot[i,4] = edges_tot[i,2]-edges_tot[i,3]
        else:
           edges_tot[i,0]=edges_tot[i-half,1]
           edges_tot[i,1]=edges_tot[i-half,0]
           edges_tot[i,2]=edges_tot[i-half,2]
           edges_tot[i,3]=weight[edges_tot[i,0].astype(int)-1,0]
           edges_tot[i,4] = edges_tot[i,2]-edges_tot[i,3]
    mst_edges=np.zeros(2*half, dtype=bool) #Array containing the activation variables of selected edges
    feasible=False
#%%  Main (until line 609 .m file)
    while go:
        flag1,flag2,flag3,flag4=True,True,True,True
        it+=1
        value_potential_edge,pos_potential_edge=np.min(edges_tot[:,4]),np.argmin(edges_tot[:,4])
        if (value_potential_edge>10**49) or (it==max_it): #Condition to stop if a C-MST cannot be found
            #print(it)
            #print(value_potential_edge)
            break
        node1,node2=edges_tot[pos_potential_edge,0].astype(int),edges_tot[pos_potential_edge,1].astype(int)
        if (CP[node1-1]==CP[node2-1]) and (flag1) and (flag2) and (flag3) and (flag4): #Condition for avoiding the creation of loops
            flag1=False  #Boolean for loops creation
            if pos_potential_edge<=half-1: #Eliminiating edges which connect the same component
                edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
            else:
                edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
    #%%  Next code is when the potential edge is connected directly to the OSS (node==1) and it does not create loops             
        if ((node1 == 1) or (node2 == 1)) and (flag1 ==True) and (flag2 ==True) and (flag3 ==True) and (flag4 ==True): #Evaluation of the number of nodes in a subtree rooted at 1
           flag2=False
           if node1==1: #If the selected edge has a node 1 the OSS
               if (S[CP[node2-1]] > UL): #Evaluation of the capacity constraint: If true, proceeding to eliminate edges 
                   if pos_potential_edge<=half-1: 
                       edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                       edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                   else:
                       edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                       edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
               else:   #If capacity constraint not violated, then evaluate no-crossing cables constraint
                    if (not(intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const))): #If no cables crossing, add the edge to the tree
                       mst_edges[pos_potential_edge]=True #Add it to the tree. line 88 .m file
                       #Update node address
                       address_nodes[node2-1]=1
                       C_sliced_n2=C[CP[node2-1]]
                       for j in range(len(C_sliced_n2)): #This could be replaced without for loop as address_nodes is now an array (before was a list)
                           if C_sliced_n2[j]==node2:
                               pass
                           else:
                               address_nodes[C_sliced_n2[j]-1]=node2
                       #Update weights and cost functions
                       if option == 1: 
                           weight[node2-1]=0
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],3]=weight[node2-1]
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],4]=edges_tot[np.where(edges_tot[:,0]==node2)[0],2]-\
                                                                          edges_tot[np.where(edges_tot[:,0]==node2)[0],3]
                       elif option == 2:
                           pass
                       elif option == 3:
                           C_sliced_n1=C[CP[node1-1]]
                           for j in range(len(C_sliced_n1)):
                               weight[C_sliced_n1[j]-1]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],4]=edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],2]-\
                                                                                       edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]                              
                       else:
                           raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option)) #Weight and cost function updated. line 126 .m file                     
                       #Eliminating selected edge from edges potential list
                       if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                       else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
                       #Updating auxiliary matrix CP, C, S
                       u,v=min(node1,node2),max(node1,node2)
                       C_sliced_u,C_sliced_v=C[CP[u-1]],C[CP[v-1]]
                       S[CP[u-1]] = len(C_sliced_u) + len(C_sliced_v) #Updating size of components
                       C[CP[u-1]]+=C[CP[v-1]] #Merging two lists due to component's merge 
                       old_pos = CP[v-1]
                       for j in range(len(C_sliced_v)): #Updating components position for each merged node
                           CP[C_sliced_v[j]-1]=CP[u-1]
                       for j in range(len(CP)):
                           if CP[j]>old_pos:
                               CP[j]-=1
                       del C[old_pos] #Deleting old component
                       del S[old_pos] #Deleting old component size (line 167 .m file)
                    else: #If a cable crossing is detected
                        if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                        else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
           if node2==1: #If the selected edge has a node 2 the OSS
              if (S[CP[node1-1]] > UL): #Evaluation of the capacity constraint: If true, proceeding to eliminate edges 
                   if pos_potential_edge<=half-1: 
                       edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                       edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                   else:
                       edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                       edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
              else:
                    if (not(intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const))): #If no cables crossing, add the edge to the tree
                       mst_edges[pos_potential_edge]=True #Add it to the tree. line 190 .m file
                       #Update node address
                       address_nodes[node1-1]=1
                       C_sliced_n1=C[CP[node1-1]]
                       for j in range(len(C_sliced_n1)):
                           if C_sliced_n1[j]==node1:
                               pass
                           else:
                               address_nodes[C_sliced_n1[j]-1]=node1 
                       #Update weights and cost functions
                       if option == 1: 
                           weight[node2-1]=0
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],3]=weight[node2-1]
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],4]=edges_tot[np.where(edges_tot[:,0]==node2)[0],2]-\
                                                                          edges_tot[np.where(edges_tot[:,0]==node2)[0],3]
                       elif option == 2:
                           pass
                       elif option == 3:
                           #C_sliced_n1=C[CP[node1-1]]
                           for j in range(len(C_sliced_n1)):
                               weight[C_sliced_n1[j]-1]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],4]=edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],2]-\
                                                                                       edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]                              
                       else:
                           raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option)) #Weight and cost function updated. line 226 .m file 
                       #Eliminating selected edge from edges potential list
                       if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                       else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50 #line 234 .m file
                       #Updating auxiliary matrix CP, C, S
                       u,v=min(node1,node2),max(node1,node2)
                       C_sliced_u,C_sliced_v=C[CP[u-1]],C[CP[v-1]]
                       S[CP[u-1]] = len(C_sliced_u) + len(C_sliced_v) #Updating size of components
                       C[CP[u-1]]+=C[CP[v-1]] #Merging two lists due to component's merge 
                       old_pos = CP[v-1]
                       for j in range(len(C_sliced_v)): #Updating components position for each merged node
                           CP[C_sliced_v[j]-1]=CP[u-1]
                       for j in range(len(CP)):
                           if CP[j]>old_pos:
                               CP[j]-=1
                       del C[old_pos] #Deleting old component
                       del S[old_pos] #Deleting old component size (line 267 .m file)
                    else: #If a cable crossing is detected
                        if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                        else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
    #%%  Next code is when the potential edge is not connected directly to the OSS (node==1) and it does not create loops. Two cases: One of the components has as element node=1 or none of them.
        if (flag1 ==True) and (flag2 ==True) and (flag3 ==True) and (flag4 ==True): 
           if (1 in C[CP[node1-1]]) or (1 in C[CP[node2-1]]): #One of the components has an element '1' (OSS)
               flag3 ==False #line 284 .m file
               if (1 in C[CP[node1-1]]): #The component of node1 includes the root 1
                  if address_nodes[node1-1]==1: #The node 1 is connected directly to the OSS (element '1')
                     tot_nodes=np.where(address_nodes==node1)[0].size+S[CP[node2-1]]+1
                  else: #The node 1 is not connected directly to the OSS (element '1')
                     tot_nodes=np.where(address_nodes==address_nodes[node1-1])[0].size+S[CP[node2-1]]+1
                  if tot_nodes>UL: ##Evaluation of the capacity constraint: If true, proceeding to eliminate edges 
                     if pos_potential_edge<=half-1: 
                        edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                        edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                     else:
                        edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                        edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
                  else:#No violation of capacity constraint
                      if (not(intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const))): #If no cables crossing, add the edge to the tree
                          mst_edges[pos_potential_edge]=True #Add it to the tree. line 301 .m file
                          #Update node address
                          if address_nodes[node1-1]==1:
                             C_sliced_n2=C[CP[node2-1]]
                             for j in range(len(C_sliced_n2)):
                                address_nodes[C_sliced_n2[j]-1]=node1
                          else:
                             C_sliced_n2=C[CP[node2-1]]
                             for j in range(len(C_sliced_n2)):
                                address_nodes[C_sliced_n2[j]-1]=address_nodes[node1-1] #line 318 .m file
                          #Update weights and cost functions
                          if option == 1: 
                             weight[node2-1]=0
                             edges_tot[np.where(edges_tot[:,0]==node2)[0],3]=weight[node2-1]
                             edges_tot[np.where(edges_tot[:,0]==node2)[0],4]=edges_tot[np.where(edges_tot[:,0]==node2)[0],2]-\
                                                                          edges_tot[np.where(edges_tot[:,0]==node2)[0],3]
                          elif option == 2:
                              pass
                          elif option == 3:
                              C_sliced_n1=C[CP[node1-1]]
                              for j in range(len(C_sliced_n1)):
                                  weight[C_sliced_n1[j]-1]=weight[node2-1]
                                  edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]=weight[node2-1]
                                  edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],4]=edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],2]-\
                                                                                       edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]                              
                          else:
                              raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option)) #Weight and cost function updated. line 344 .m file
                          #Eliminating selected edge from edges potential list
                          if pos_potential_edge<=half-1: 
                             edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                             edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                          else:
                             edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                             edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50 #line 352 .m file
                          #Updating auxiliary matrix CP, C, S
                          u,v=min(node1,node2),max(node1,node2)
                          C_sliced_u,C_sliced_v=C[CP[u-1]],C[CP[v-1]]
                          S[CP[u-1]] = len(C_sliced_u) + len(C_sliced_v) #Updating size of components
                          C[CP[u-1]]+=C[CP[v-1]] #Merging two lists due to component's merge 
                          old_pos = CP[v-1]
                          for j in range(len(C_sliced_v)): #Updating components position for each merged node
                              CP[C_sliced_v[j]-1]=CP[u-1]
                          for j in range(len(CP)):
                              if CP[j]>old_pos:
                                 CP[j]-=1
                          del C[old_pos] #Deleting old component
                          del S[old_pos] #Deleting old component size (line 385 .m file)
                      else:#If a cable crossing is detected
                          if pos_potential_edge<=half-1: 
                              edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                              edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                          else:
                              edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                              edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50  #(line 396 .m file)
               else: #The component of node2 includes the root 1
                  if address_nodes[node2-1]==1: #The node 2 is connected directly to the OSS (element '1')
                     tot_nodes=np.where(address_nodes==node2)[0].size+S[CP[node1-1]]+1
                  else: #The node 2 is not connected directly to the OSS (element '1')
                     tot_nodes=np.where(address_nodes==address_nodes[node2-1])[0].size+S[CP[node1-1]]+1
                  if tot_nodes>UL: ##Evaluation of the capacity constraint: If true, proceeding to eliminate edges 
                     if pos_potential_edge<=half-1: 
                        edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                        edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                     else:
                        edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                        edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
                  else: #No violation of capacity constraint
                      if (not(intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const))): #If no cables crossing, add the edge to the tree
                          mst_edges[pos_potential_edge]=True #Add it to the tree. line 413 .m file
                          #Update node address
                          if address_nodes[node2-1]==1:
                             C_sliced_n1=C[CP[node1-1]]
                             for j in range(len(C_sliced_n1)):
                                address_nodes[C_sliced_n1[j]-1]=node2
                          else:
                             C_sliced_n1=C[CP[node1-1]]
                             for j in range(len(C_sliced_n1)):
                                address_nodes[C_sliced_n1[j]-1]=address_nodes[node2-1] #line 430 .m file
                          #Update weights and cost functions
                          if option == 1: 
                             weight[node2-1]=0
                             edges_tot[np.where(edges_tot[:,0]==node2)[0],3]=weight[node2-1]
                             edges_tot[np.where(edges_tot[:,0]==node2)[0],4]=edges_tot[np.where(edges_tot[:,0]==node2)[0],2]-\
                                                                          edges_tot[np.where(edges_tot[:,0]==node2)[0],3]
                          elif option == 2:
                              pass
                          elif option == 3:
                              C_sliced_n1=C[CP[node1-1]]
                              for j in range(len(C_sliced_n1)):
                                  weight[C_sliced_n1[j]-1]=weight[node2-1]
                                  edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]=weight[node2-1]
                                  edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],4]=edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],2]-\
                                                                                       edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]                              
                          else:
                              raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option)) #Weight and cost function updated. line 456 .m file
                          #Eliminating selected edge from edges potential list
                          if pos_potential_edge<=half-1: 
                             edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                             edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                          else:
                             edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                             edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50 #line 464 .m file
                          #Updating auxiliary matrix CP, C, S
                          u,v=min(node1,node2),max(node1,node2)
                          C_sliced_u,C_sliced_v=C[CP[u-1]],C[CP[v-1]]
                          S[CP[u-1]] = len(C_sliced_u) + len(C_sliced_v) #Updating size of components
                          C[CP[u-1]]+=C[CP[v-1]] #Merging two lists due to component's merge 
                          old_pos = CP[v-1]
                          for j in range(len(C_sliced_v)): #Updating components position for each merged node
                              CP[C_sliced_v[j]-1]=CP[u-1]
                          for j in range(len(CP)):
                              if CP[j]>old_pos:
                                 CP[j]-=1
                          del C[old_pos] #Deleting old component
                          del S[old_pos] #Deleting old component size (line 497 .m file)
                      else: #If a cable crossing is detected
                          if pos_potential_edge<=half-1: 
                              edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                              edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                          else:
                              edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                              edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50  #(line 507 .m file)
           else:  # Node of the components has as element '1' (OSS)
               flag4=False
               if (S[CP[node1-1]]+S[CP[node2-1]]> UL): #Evaluation of the capacity constraint: If true, proceeding to eliminate edges 
                  if pos_potential_edge<=half-1: 
                     edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                     edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                  else:
                     edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                     edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
               else: #If no violation of the capacity constraint
                    if (not(intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const))): #If no cables crossing, add the edge to the tree
                       mst_edges[pos_potential_edge]=True #Add it to the tree. line 522 .m file
                       #Update weights and cost functions
                       if option == 1: 
                           weight[node2-1]=0
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],3]=weight[node2-1]
                           edges_tot[np.where(edges_tot[:,0]==node2)[0],4]=edges_tot[np.where(edges_tot[:,0]==node2)[0],2]-\
                                                                          edges_tot[np.where(edges_tot[:,0]==node2)[0],3]
                       elif option == 2:
                           pass
                       elif option == 3:
                           C_sliced_n1=C[CP[node1-1]]
                           for j in range(len(C_sliced_n1)):
                               weight[C_sliced_n1[j]-1]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]=weight[node2-1]
                               edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],4]=edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],2]-\
                                                                                       edges_tot[np.where(edges_tot[:,0]==C_sliced_n1[j])[0],3]                              
                       else:
                           raise Exception('option should be either 1, 2 or 3 The value of x was: {}'.format(option)) #Weight and cost function updated. line 548 .m file                     
                       #Eliminating selected edge from edges potential list
                       if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                       else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
                       #Updating auxiliary matrix CP, C, S
                       u,v=min(node1,node2),max(node1,node2)
                       C_sliced_u,C_sliced_v=C[CP[u-1]],C[CP[v-1]]
                       S[CP[u-1]] = len(C_sliced_u) + len(C_sliced_v) #Updating size of components
                       C[CP[u-1]]+=C[CP[v-1]] #Merging two lists due to component's merge 
                       old_pos = CP[v-1]
                       for j in range(len(C_sliced_v)): #Updating components position for each merged node
                           CP[C_sliced_v[j]-1]=CP[u-1]
                       for j in range(len(CP)):
                           if CP[j]>old_pos:
                               CP[j]-=1
                       del C[old_pos] #Deleting old component
                       del S[old_pos] #Deleting old component size (line 590 .m file)
                    else: #If a cable crossing is detected
                        if pos_potential_edge<=half-1: 
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge+half,4] = edges_tot[pos_potential_edge+half,2] + 10**50
                        else:
                           edges_tot[pos_potential_edge,4] = edges_tot[pos_potential_edge,2] + 10**50
                           edges_tot[pos_potential_edge-half,4] = edges_tot[pos_potential_edge-half,2] + 10**50
        if len(C)==1:
           go=False #(line 606 .m file)
           feasible=True
    T=np.zeros((len(np.where(mst_edges==True)[0]),3))
    T[:,0]=edges_tot[np.where(mst_edges==True)[0],0]
    T[:,1]=edges_tot[np.where(mst_edges==True)[0],1]
    T[:,2]=edges_tot[np.where(mst_edges==True)[0],2]    
#%%  Running the function          
    return T,feasible         
if __name__ == "__main__": 
    #First X,Y are for artifitial wind farm Diagonal 10 (10 WTs + 1 OSS)         
    X=[10000,8285.86867841725,9097.62726321941,8736.91009361509,9548.66867841725,10360.4272632194,9187.95150881294,9999.71009361510,10811.4686784173,10450.7515088129,11262.5100936151]
    Y=[10000,9426,8278,10574,9426,8278,11722,10574,9426,11722,10574]
    #Next X,Y are for real-world wind farm Ormonde (30 WTs + 1 OSS)
    #X=[473095,471790,471394,470998,470602,470207,469811,472523,469415,472132,471742,471351,470960,470569,470179,469788,472866,472480,472094,471708,471322,470937,470551,473594,473213,472833,472452,472071,471691,471310,470929]
    #Y=[5992345,5991544,5991899,5992252,5992607,5992960,5993315,5991874,5993668,5992236,5992598,5992960,5993322,5993684,5994047,5994409,5992565,5992935,5993306,5993675,5994045,5994416,5994786,5992885,5993264,5993643,5994021,5994400,5994779,5995156,5995535]
    #Next X, Y are for real-world wind farm DanTysk (80 WTs + 1 OSS)
    #X=[387100,383400,383400,383900,383200,383200,383200,383200,383200,383200,383200,383200,383300,384200,384200,384100,384000,383800,383700,383600,383500,383400,383600,384600,385400,386000,386100,386200,386300,386500,386600,386700,386800,386900,387000,387100,387200,383900,387400,387500,387600,387800,387900,388000,387600,386800,385900,385000,384100,384500,384800,385000,385100,385200,385400,385500,385700,385800,385900,385900,385500,385500,386000,386200,386200,384500,386200,386700,386700,386700,384300,384400,384500,384600,384300,384700,384700,384700,385500,384300,384300]
    #Y=[6109500,6103800,6104700,6105500,6106700,6107800,6108600,6109500,6110500,6111500,6112400,6113400,6114000,6114200,6115100,6115900,6116700,6118400,6119200,6120000,6120800,6121800,6122400,6122000,6121700,6121000,6120000,6119100,6118100,6117200,6116200,6115300,6114300,6113400,6112400,6111500,6110700,6117600,6108900,6108100,6107400,6106300,6105200,6104400,6103600,6103600,6103500,6103400,6103400,6104400,6120400,6119500,6118400,6117400,6116500,6115500,6114600,6113500,6112500,6111500,6105400,6104200,6110400,6109400,6108400,6121300,6107500,6106400,6105300,6104400,6113300,6112500,6111600,6110800,6110100,6109200,6108400,6107600,6106500,6106600,6105000]
    X=np.array(X)
    Y=np.array(Y)    
    option=3
    UL=15
    Inters_const=True
    T,feasible=capacitated_spanning_tree(X,Y,option,UL,Inters_const)

    print("The total length of the solution is {value:.2f} m".format(value = sum(T[:,2])))
    print("Feasibility: {feasible1}".format(feasible1=feasible))
    plt.figure()
    plt.plot(X[1:], Y[1:], 'r+',markersize=10, label='Turbines')
    plt.plot(X[0], Y[0], 'ro',markersize=10, label='OSS')
    for i in range(len(X)):
        plt.text(X[i]+50, Y[i]+50,str(i+1))
    n1xs = X[T[:,0].astype(int)-1].ravel().T
    n2xs = X[T[:,1].astype(int)-1].ravel().T
    n1ys = Y[T[:,0].astype(int)-1].ravel().T
    n2ys = Y[T[:,1].astype(int)-1].ravel().T
    xs = np.vstack([n1xs,n2xs])
    ys = np.vstack([n1ys,n2ys])
    plt.plot(xs,ys,'b')
    plt.legend()