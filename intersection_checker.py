# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:08:54 2020

@author: juru
"""

import numpy as np
from two_lines_intersecting import two_lines_intersecting

def intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y,Inters_const): 
    current_edges=np.where(mst_edges==True)[0]
    current_edges_size=current_edges.size
    intersection=False
    if Inters_const:
        if current_edges_size==0:
            pass
        else:
            for i in range(current_edges_size):
                line1=np.array([[X[edges_tot[pos_potential_edge,0].astype(int)-1],Y[edges_tot[pos_potential_edge,0].astype(int)-1]],\
                                [X[edges_tot[pos_potential_edge,1].astype(int)-1],Y[edges_tot[pos_potential_edge,1].astype(int)-1]]])
                line2=np.array([[X[edges_tot[current_edges[i],0].astype(int)-1],Y[edges_tot[current_edges[i],0].astype(int)-1]],\
                                [X[edges_tot[current_edges[i],1].astype(int)-1],Y[edges_tot[current_edges[i],1].astype(int)-1]]])
                if two_lines_intersecting(line1,line2):
                    intersection=True
                    break
    return intersection

#if __name__ == '__main__':
    #cross=intersection_checker(pos_potential_edge,edges_tot,mst_edges,X,Y)
