# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:30:40 2020

@author: juru
"""
import numpy as np

def var(WTc,OSSc,X,Y,UL,Cables,T,feasibility):
#%% Obtaining edges_tot matrix    
    n_wt_oss=WTc+OSSc  #Defining number of wind turbines with OSS
    half=int(n_wt_oss*(n_wt_oss-1)/2)  
    edges_tot=np.zeros((half,3)) #Defining the matrix with Edges information
    cont_edges=0
    for i in range(n_wt_oss):
        for j in range(i+1,n_wt_oss):
            edges_tot[cont_edges,0]=i+1 #First element is first node (Element =1 is the OSS. and from OSS to Nwt the WTs)
            edges_tot[cont_edges,1]=j+1 #Second element is second node
            edges_tot[cont_edges,2]=np.sqrt((X[j]-X[i])**2+(Y[j]-Y[i])**2) #Third element is the length of the edge
            cont_edges+=1
#%% Obtaining var_output
#%% OSSs             
    var_output=np.zeros((OSSc,4))
    for i in range(OSSc):
        OSSa,OSSb=np.where(edges_tot[:,0]==i+1)[0],np.where(edges_tot[:,1]==i+1)[0]
        for j in range(len(OSSa)):
            if edges_tot[OSSa[j],1]>OSSc:
                var_output=np.vstack((var_output,np.array([i+1,edges_tot[OSSa[j],1],edges_tot[OSSa[j],2],0])))
                var_output=np.vstack((var_output,np.hstack((np.full((UL,1),i+1),np.full((UL,1),edges_tot[OSSa[j],1]),
                           np.full((UL,1),edges_tot[OSSa[j],2]),np.array(range(1,UL+1)).reshape(UL,1)))))
        for j in range(len(OSSb)):
            if edges_tot[OSSb[j],0]>OSSc:
                var_output=np.vstack((var_output,np.array([i+1,edges_tot[OSSb[j],0],edges_tot[OSSb[j],2],0])))
                var_output=np.vstack((var_output,np.hstack((np.full((UL,1),i+1),np.full((UL,1),edges_tot[OSSb[j],0]),
                           np.full((UL,1),edges_tot[OSSb[j],2]),np.array(range(1,UL+1)).reshape(UL,1)))))
#%% WTs                 
    for i in range(OSSc+1,OSSc+WTc+1):
        WTa,WTb,var_output_aux=np.where(edges_tot[:,0]==i)[0],np.where(edges_tot[:,1]==i)[0],np.zeros((0,3))
        for j in range(len(WTa)):
            if edges_tot[WTa[j],1]>OSSc:
                var_output_aux=np.vstack((var_output_aux,np.array([i,edges_tot[WTa[j],1],edges_tot[WTa[j],2]])))
                #var_output_aux=np.vstack((var_output_aux,np.hstack((np.full((UL,1),i+1),np.full((UL,1),edges_tot[OSSa[j],1]),
                           #np.full((UL,1),edges_tot[OSSa[j],2]),np.array(range(1,UL+1)).reshape(UL,1)))))
        for j in range(len(WTb)):
            if edges_tot[WTb[j],0]>OSSc:
                var_output_aux=np.vstack((var_output_aux,np.array([i,edges_tot[WTb[j],0],edges_tot[WTb[j],2]])))
                #var_output=np.vstack((var_output,np.hstack((np.full((UL,1),i+1),np.full((UL,1),edges_tot[OSSb[j],0]),
                           #np.full((UL,1),edges_tot[OSSb[j],2]),np.array(range(1,UL+1)).reshape(UL,1)))))
        var_output_aux = var_output_aux[np.argsort(var_output_aux[:,2])]
        for k in range(T):
            for p in range(UL):
                var_output=np.vstack((var_output,np.hstack((var_output_aux[k,:],p))))                            
#%% Cost coefficients calculation
    rows_var,rows_cables=var_output.shape[0],Cables.shape[0]
    objective,cable_selected=np.zeros((rows_var,1)),np.zeros((rows_var,1))
    for i in range(rows_var):
        if var_output[i,3]!=0:
           for j in range(rows_cables):
               if var_output[i,3]<=Cables[j,1]:
                  cable_selected[i]=j
                  if not(feasibility):
                     objective[i]=Cables[j,2]*(var_output[i,2])/1000 
                  else:
                      objective[i]=0
                  break
                
    return var_output,objective,cable_selected
if __name__ == "__main__":
    WTc=10
    OSSc=1
    X=[10000,8285.86867841725,9097.62726321941,8736.91009361509,9548.66867841725,10360.4272632194,9187.95150881294,9999.71009361510,10811.4686784173,10450.7515088129,11262.5100936151]
    Y=[10000,9426,8278,10574,9426,8278,11722,10574,9426,11722,10574]  
    UL=6
    Cables=np.array([[500,2,100000],[800,4,150000],[1000,6,250000]])
    T=5
    feasibility=0
    var_output,objective,cable_selected=var(WTc,OSSc,X,Y,UL,Cables,T,feasibility)

    