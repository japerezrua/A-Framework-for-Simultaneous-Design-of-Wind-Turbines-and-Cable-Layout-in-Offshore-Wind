# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:17:52 2020

@author: juru
"""

import topfarm
import time
#import sys
#import w2l
#import workshop
#import os
# non-updating, inline plots
#%matplotlib inline
# ...or updating plots in new window
#%matplotlib
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.easy_drivers import EasyRandomSearchDriver, EasyScipyOptimizeDriver, EasySimpleGADriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
#from topfarm.cost_models.electrical.simple_msp import ElNetLength, ElNetCost 
#import matplotlib.pylab as plt
from global_collection import global_optimizer
from plotting_global import plotting

# Convenient plot component for plotting the cables from simple_msp in jupyter notebooks
from plotting_cmst import XYCablePlotComp

#%%  Setting up the site (IEA-37 site) to optimize and the Wind Turbine
from workshop.cabling import get_site
from py_wake.examples.data.DTU10MW_RWT import DTU10MW

t = time.time()
site = get_site()
n_wt = len(site.initial_position)
windTurbines = DTU10MW()
Drotor_vector = [windTurbines.diameter()] * n_wt 
power_rated_vector = [float(windTurbines.power(20)/1000)] * n_wt 
hub_height_vector = [windTurbines.hub_height()] * n_wt 
rated_rpm_array = 12. * np.ones([n_wt])

print('Number of turbines:', n_wt)

#%%  Setting up the AEP calculator: Using Gaussian wake model from Bastankhah. Based on 16 wind directions
#    It should not be critical as the RandomSearch algorithm will be implemented

from py_wake.wake_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.aep_calculator import AEPCalculator

## We use the Gaussian wake model
wake_model = IEA37SimpleBastankhahGaussian(site, windTurbines)
AEPCalc = AEPCalculator(wake_model)

## The AEP is calculated using n_wd wind directions
n_wd = 16
wind_directions = np.linspace(0., 360., n_wd, endpoint=False)

def aep_func(x, y, **kwargs):
    """A simple function that takes as input the x,y position of the turbines and return the AEP"""
    return AEPCalc.calculate_AEP(x_i=x, y_i=y, wd=wind_directions).sum(-1).sum(-1)*10**6
#%% Setting up the NREL IRR cost model (Based on the 2006 NREL report)
from topfarm.cost_models.economic_models.turbine_cost import economic_evaluation as EE_NREL

def irr_nrel(aep, electrical_connection_cost, **kwargs):
    return EE_NREL(Drotor_vector, power_rated_vector, hub_height_vector, aep, electrical_connection_cost).calculate_irr()
#%% Setting up the DTU IRR cost model
from topfarm.cost_models.economic_models.dtu_wind_cm_main import economic_evaluation as EE_DTU

distance_from_shore = 10.0 # [km]
energy_price = 0.2 / 7.4 # [DKK/kWh] / [DKK/EUR] -> [EUR/kWh]
project_duration = 20 # [years]
water_depth_array = 20 * np.ones([n_wt])
Power_rated_array = np.array(power_rated_vector)/1.0E3 # [MW]

ee_dtu = EE_DTU(distance_from_shore, energy_price, project_duration)


def irr_dtu(aep, electrical_connection_cost, **kwargs):
    #print (rated_rpm_array,Drotor_vector,Power_rated_array,hub_height_vector,water_depth_array, aep,electrical_connection_cost)
    ee_dtu.calculate_irr(
                    rated_rpm_array, 
                    Drotor_vector, 
                    Power_rated_array,
                    hub_height_vector, 
                    water_depth_array, 
                    aep, 
                    electrical_connection_cost)
    return ee_dtu.IRR
#%% Setting up the Topfarm problem
## Some user options
#@markdown Which IRR Cost model to use
IRR_COST = 'DTU' #@param ["DTU", "NREL"]
driver='Random'# Random, SLSQP, Cobyla, GA
iterations=10000 #Natural number
max_time=10*3600 # Seconds

option=3
Inters_const=False
max_it=1000000
plot=False
UL=6
Cables=np.array([[500,2,100000],[800,4,150000],[1000,6,250000]])
CablesForGlobal=Cables
ULForCables=UL

#option=3
#UL=25
#Inters_const=False
#max_it=1000000
#Cables=np.array([[500,25,0.0001]])
#plot=False

dis_south=2000.0
centroid_oss=True
#@markdown Minimum spacing between the turbines
min_spacing = 2 #@param {type:"slider", min:2, max:10, step:1}

#@markdown Minimum spacing between the turbines
#cable_cost_per_meter = 750. #@param {type:"slider", min:0, max:10000, step:1}

## Electrical grid cable components (Minimum spanning tree from Topfarm report 2010)
#elnetlength = ElNetLength(n_wt=n_wt)
#elnetcost = ElNetCost(n_wt=n_wt, output_key='electrical_connection_cost', cost_per_meter=cable_cost_per_meter)'

from collection_system import collection_system
import openmdao.api as om
#cost_vectorized=np.zeros(50)
#locs=np.zeros((50,n_wt,2))
#for i in range(50): 
class ElNetCost(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('option', types=int)
        self.options.declare('UL', types=int)
        self.options.declare('Inters_const',types=bool)
        self.options.declare('max_it', types=int)
        self.options.declare('Cables',types=np.ndarray)
        self.options.declare('plot',types=bool)

    def setup(self):
        self.add_input(topfarm.x_key, np.zeros(self.options['n_wt']))
        self.add_input(topfarm.y_key, np.zeros(self.options['n_wt']))
        self.add_input('oss_location',np.zeros(2))
        self.add_output('electrical_connection_cost', 0.0)

    def compute(self, inputs, outputs):
        x,y,oss= inputs[topfarm.x_key], inputs[topfarm.y_key], inputs['oss_location'] 
        x=np.insert(x,0,oss[0])
        y=np.insert(y,0,oss[1])
        T,elnet_layout = collection_system(x,y,self.options['option'],self.options['UL'],self.options['Inters_const'] \
                                         ,self.options['max_it'],self.options['Cables'],self.options['plot'])
        outputs['electrical_connection_cost'] = elnet_layout
class Substation(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt',types=int)
        self.options.declare('south',types=float)
        self.options.declare('abscissa',types=np.ndarray)
        self.options.declare('dis_south',types=float)
        self.options.declare('centroid_oss',types=bool)       
    def setup(self):
        self.add_input(topfarm.x_key,np.zeros(self.options['n_wt']))
        self.add_input(topfarm.y_key,np.zeros(self.options['n_wt']))
        self.add_output('oss_location',np.zeros(2))
    def compute(self,inputs,outputs):
        x, y = inputs[topfarm.x_key], inputs[topfarm.y_key]
        if self.options['centroid_oss']:
           oss=np.zeros(2)
           oss[0],oss[1]=sum(x)/len(x),sum(y)/len(y)
           outputs['oss_location']=oss
        else:
           oss=np.zeros(2)
           oss[0],oss[1]=np.average(self.options['abscissa']),self.options['south']-self.options['dis_south']
           outputs['oss_location']=oss           
        
elnetcost = ElNetCost(n_wt=n_wt,option=option,UL=UL,Inters_const=Inters_const,max_it=max_it,Cables=Cables,plot=plot)
substation=Substation(n_wt=n_wt,south=min(site.boundary[:,1]),abscissa=np.array([min(site.boundary[:,0]),max(site.boundary[:,0])]),\
                     dis_south=dis_south,centroid_oss=centroid_oss)

# The Topfarm IRR cost model components
irr_dtu_comp = CostModelComponent(input_keys=['aep', ('electrical_connection_cost', 0.0)],   n_wt=n_wt, 
                                  cost_function=irr_dtu, output_key="irr", output_unit="%",   objective=True, income_model=True)
irr_nrel_comp = CostModelComponent(input_keys=['aep', ('electrical_connection_cost', 0.0)],   n_wt=n_wt, 
                                   cost_function=irr_nrel, output_key="irr", output_unit="%",   objective=True, income_model=True)
irr_cost_models = {'DTU': irr_dtu_comp, 'NREL': irr_nrel_comp}

driver_r=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=iterations,max_time=max_time)
driver_slsqp=EasyScipyOptimizeDriver(optimizer='SLSQP', maxiter=iterations, tol=1e-6, disp=False)
driver_cobyla=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=iterations, tol=1e-6, disp=False)
driver_ga=EasySimpleGADriver(max_gen=iterations, pop_size=5, Pm=0.3, Pc=.5, elitism=True, bits={})

driver_op={'Random':driver_r, 'SLSQP':driver_slsqp, 'Cobyla':driver_cobyla, 'GA':driver_ga}


## The Topfarm AEP component
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, 
                              output_key="aep", output_unit="GWh", objective=False, output_val=np.zeros(n_wt))

## Plotting component
plot_comp = XYCablePlotComp(memory=0, plot_improvements_only=False, plot_initial=False,option=option,UL=UL,\
                            Inters_const=Inters_const,max_it=max_it,Cables=Cables,plot=plot,south=min(site.boundary[:,1]),\
                            north=max(site.boundary[:,1]),abscissa=np.array([min(site.boundary[:,0]),max(site.boundary[:,0])]),\
                            dis_south=dis_south,centroid_oss=centroid_oss)


## The group containing all the components
#group = TopFarmGroup([aep_comp, elnetlength, elnetcost, irr_cost_models[IRR_COST]])
group = TopFarmGroup([aep_comp, substation, elnetcost, irr_cost_models[IRR_COST]])

def optimize(driver):
    problem = TopFarmProblem(design_vars=dict(zip('xy', site.initial_position.T)),cost_comp=group,\
    driver=driver_op[driver],constraints=[SpacingConstraint(min_spacing * windTurbines.diameter(0)),\
    XYBoundaryConstraint(site.boundary)],expected_cost=1.0,plot_comp=plot_comp)
    cost, state, recorder = problem.optimize()
    return cost, state, recorder
   
cost, state, recorder=optimize(driver)
elapsed = time.time() - t

if centroid_oss:
   oss=np.zeros(2)
   oss[0],oss[1]=sum(state['x'])/len(state['x']),sum(state['y'])/len(state['y'])
else:
   oss=np.zeros(2)
   abscissa=np.array([min(site.boundary[:,0]),max(site.boundary[:,0])])
   south=min(site.boundary[:,1])
   oss[0],oss[1]=np.average(abscissa),south-dis_south

X=np.insert(state['x'],0,oss[0])
Y=np.insert(state['y'],0,oss[1])

Tree,Cost_collection=global_optimizer(WTc=n_wt,X=X,Y=Y,Cables=CablesForGlobal)
plotting(X,Y,CablesForGlobal,Tree)

aep_final=AEPCalc.calculate_AEP(x_i=state['x'], y_i=state['y'], wd=wind_directions).sum(-1).sum(-1)*10**6
print("Solution cost with global optimizer:",ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, Cost_collection))

T2,elnet_layout2 = collection_system(X,Y,option,ULForCables,Inters_const \
                                   ,max_it,CablesForGlobal,True)

print("Solution cost with heuristic:",ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, elnet_layout2))
#plt.close()
#print(i)
#cost_vectorized[i]=cost
#locs[i,:,0]=state['x']
#locs[i,:,1]=state['y']

#plt.plot(-cost_vectorized)
