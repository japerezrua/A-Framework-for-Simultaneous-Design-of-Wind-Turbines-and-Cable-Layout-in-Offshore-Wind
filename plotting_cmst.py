# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:27:33 2020

@author: juru
"""

from topfarm.cost_models.utils.spanning_tree import spanning_tree
from IPython.display import clear_output
import matplotlib.pylab as plt
from plot_class import XYPlotComp
from py_wake.site import UniformWeibullSite
from py_wake.site.shear import PowerShear
import numpy as np
from collection_system import collection_system

class XYCablePlotComp(XYPlotComp):
    def __init__(self,memory,plot_improvements_only,plot_initial,option,UL,Inters_const,max_it,Cables,plot,south,north,abscissa,\
                 dis_south,centroid_oss):
        self.option=option
        self.Inters_const=Inters_const
        self.max_it=max_it
        self.Cables=Cables
        self.plot=plot
        self.UL=UL
        self.south=south
        self.north=north
        self.abscissa=abscissa
        self.dis_south=dis_south
        self.centroid_oss=centroid_oss
        super().__init__(memory,plot_improvements_only,plot_initial)
    def plot_current_position(self, x, y):

        XYPlotComp.plot_current_position(self, x, y)

        if self.centroid_oss:
           oss=np.zeros(2)
           oss[0],oss[1]=sum(x)/len(x),sum(y)/len(y)
        else:
           oss=np.zeros(2)
           oss[0],oss[1]=np.average(self.abscissa),self.south-self.dis_south
        plt.plot(oss[0],oss[1],'bo',markersize=8,label='OSS')
        x=np.insert(x,0,oss[0])
        y=np.insert(y,0,oss[1])
        T,amount = collection_system(x, y,self.option,self.UL,self.Inters_const,self.max_it,self.Cables,self.plot)
        #indices = (T[:,0:2]-1).T.astype(int)        
        #plt.plot(x[indices], y[indices], color='r')
        if not(self.centroid_oss):
           plt.ylim(1.2*oss[1], 1.1*self.north)
        colors = ['b','g','r','c','m','y','k','bg','gr','rc','cm']
        for i in range(self.Cables.shape[0]):
            indices = T[:,3]==i
            if indices.any():
               n1xs = x[T[indices,0].astype(int)-1].ravel().T
               n2xs = x[T[indices,1].astype(int)-1].ravel().T
               n1ys = y[T[indices,0].astype(int)-1].ravel().T
               n2ys = y[T[indices,1].astype(int)-1].ravel().T
               xs = np.vstack([n1xs,n2xs])
               ys = np.vstack([n1ys,n2ys])
               plt.plot(xs,ys,'{}'.format(colors[i]))
               plt.plot([],[],'{}'.format(colors[i]),label='Cable: {} mm2'.format(self.Cables[i,0]))
        plt.legend()
    def compute(self, inputs, outputs):
        clear_output(wait=True)
        XYPlotComp.compute(self, inputs, outputs)
        plt.show()
        
