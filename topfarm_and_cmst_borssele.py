# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 13:17:52 2020

@author: juru
"""
import topfarm
import time
t = time.time()
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
from cables_cost_function import cost_function_array

# Convenient plot component for plotting the cables from simple_msp in jupyter notebooks
from plotting_cmst import XYCablePlotComp


#%%  Setting up the site (IEA-37 Borssele site) to optimize and the Wind Turbine
from borssele import get_site
from borssele import IEA10MW

site = get_site()
n_wt = len(site.initial_position)
windTurbines = IEA10MW()
wt_power=10000 #Wind turbine power in kilowatt
Drotor_vector = [windTurbines.diameter()] * n_wt 
power_rated_vector = wt_power* np.ones(n_wt)
hub_height_vector = [windTurbines.hub_height()] * n_wt 
rated_rpm_array = 8.68 * np.ones([n_wt])

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
    ee_dtu.calculate_irr(
                    rated_rpm_array, 
                    Drotor_vector, 
                    Power_rated_array,
                    hub_height_vector, 
                    water_depth_array, 
                    aep, 
                    electrical_connection_cost)
    return ee_dtu.IRR
#%% Other inputs
## Some user options
#@markdown Which IRR Cost model to use
IRR_COST = 'DTU' #@param ["DTU", "NREL"]
driver='Random'# Random, SLSQP, Cobyla, GA
iterations=150000 #Natural number. Iterations of the main driver.
max_time=0.001*3600 # Seconds. Maximum time for the main driver
optimized_with_cables=False #Activate if desired to optimize wind turbines locations with cable layout design
                           #If False, then zero cost of length? By default is zero cost

voltage=33 #Voltage level in kV of the collection system (33 kV, 66 kV)
C=10 #Maximum number of feeders connected to the offshore substations

option=3 #Choose heuristic for the cable layout. 1 = Prim. 2 = Kruskal. 3 = Esau-Williams
Inters_const=False #Activate non-cables crossing
max_it=1000000 #Maximum number of iterations of the heuristics
plot=False #Activate plotting of the cable layout every iteration

Cables=np.array([[500,4],[800,6],[1000,8]]) #Set of cables available for the full problem

centroid_oss=True #Location of the offshore substation at the centroid of the wind turbins
Correction=True #In case centroid_oss=True, do you want to correct the OSS position for no overlapping with WTs?
oss_coord=np.array([((site.boundary[0,0]+site.boundary[7,0])/2)-500,((site.boundary[0,1]+site.boundary[7,1])/2)-500])  #If centroid_oss=False, provide OSS coordinates
#@markdown Minimum spacing between the turbines
min_spacing = 2 #@param {type:"slider", min:2, max:10, step:1}

#%% Setting up and running the Topfarm problem
from collection_system import collection_system
from substation import substation_location
import openmdao.api as om
#cost_vectorized=np.zeros(50)
#locs=np.zeros((50,n_wt,2))
#for i in range(50): 
Cables=cost_function_array(wt_power,voltage,Cables) 
CablesForGlobal=Cables 
ULForCables=max(CablesForGlobal[:,1]).item()
if not(optimized_with_cables): Cables=np.array([[float('nan'),n_wt,0.00001]]) 
UL=int(max(Cables[:,1]).item())

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
        #elnet_layout=50000000       
        outputs['electrical_connection_cost'] = elnet_layout 

class Substation(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt',types=int)
        self.options.declare('oss_coord',types=np.ndarray)
        self.options.declare('centroid_oss',types=bool)
        self.options.declare('Diameter',types=float)
        self.options.declare('Correction',types=bool)
    def setup(self):
        self.add_input(topfarm.x_key,np.zeros(self.options['n_wt']))
        self.add_input(topfarm.y_key,np.zeros(self.options['n_wt']))
        self.add_output('oss_location',np.zeros(2))
    def compute(self,inputs,outputs):
        x, y = inputs[topfarm.x_key], inputs[topfarm.y_key]
        if self.options['centroid_oss']:
           #oss=np.zeros(2)
           #oss[0],oss[1]=sum(x)/len(x),sum(y)/len(y)
           oss=substation_location(self.options['n_wt'],x,y,self.options['Diameter'],self.options['Correction'])
           outputs['oss_location']=oss
        else:
           oss=self.options['oss_coord']
           outputs['oss_location']=oss           
        
elnetcost = ElNetCost(n_wt=n_wt,option=option,UL=UL,Inters_const=Inters_const,max_it=max_it,Cables=Cables,plot=plot)
substation=Substation(n_wt=n_wt,oss_coord=oss_coord,centroid_oss=centroid_oss,Diameter=windTurbines.diameter(),Correction=Correction)

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

## Plotting component (Needs to be modified to support correction and new oss fixed location)
#plot_comp = XYCablePlotComp(memory=0, plot_improvements_only=True, plot_initial=False,option=option,UL=UL,\
                           # Inters_const=Inters_const,max_it=max_it,Cables=Cables,plot=plot,south=min(site.boundary[:,1]),\
                            #centroid_oss=centroid_oss)


## The group containing all the components
#group = TopFarmGroup([aep_comp, elnetlength, elnetcost, irr_cost_models[IRR_COST]])
group = TopFarmGroup([aep_comp, substation, elnetcost, irr_cost_models[IRR_COST]])

# problem for optimization
problem = TopFarmProblem(design_vars=dict(zip('xy', site.initial_position.T)),cost_comp=group,\
driver=driver_op[driver],constraints=[SpacingConstraint(min_spacing * windTurbines.diameter(0)),\
                                      
#Choose whether to include insitu plotting or not                               
#XYBoundaryConstraint(site.boundary)],expected_cost=1.0,plot_comp=plot_comp)
XYBoundaryConstraint(site.boundary)],expected_cost=1.0)

#testing Borselle i/o    
#problem.run_once()    
#problem.cost_comp.list_outputs()
#problem.cost_comp.list_inputs()

# Run optimization if desired
cost, state, recorder = problem.optimize()
elapsed1 = time.time() - t

#%% Running global optimizer for cable layout
if centroid_oss:
   oss=substation_location(n_wt=n_wt,X=state['x'],Y=state['y'],Diameter=windTurbines.diameter(),Correction=Correction)
else:
   oss=oss_coord
X=np.insert(state['x'],0,oss[0])
Y=np.insert(state['y'],0,oss[1])
t = time.time()
Tree,Cost_collection,gap_outputit,time_formulating,time_solving,solutions=global_optimizer(WTc=n_wt,X=X,Y=Y,
Cables=CablesForGlobal,C=C)
plotting(X,Y,CablesForGlobal,Tree)
elapsed2 = time.time() - t

T2,elnet_layout2 = collection_system(X,Y,option,ULForCables,Inters_const,max_it,CablesForGlobal,True)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(-np.array(recorder.driver_iteration_dict['cost']))

aep_final=AEPCalc.calculate_AEP(x_i=state['x'], y_i=state['y'], wd=wind_directions).sum(-1).sum(-1)*10**6

#%% Printing solution
print("Solution cost of the objective function:", -cost)

print("Objective cost with heuristic using full set of cables available:",ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, elnet_layout2))

print("Objective cost with global optimizer using full set of cables available:",ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, Cost_collection))

print("AEP in GWh:",sum(aep_final)/(10**6))
print("Cost of cable layout in MEuro of the global optimizer:",Cost_collection/(10**6))
print("Cost of cable layout in MEuro of the heuristic:",elnet_layout2/(10**6))

print("Computing time of the main driver in hours:",elapsed1/3600)
print("Computing time of the global optimizer in minutes:",elapsed2/60)

print("Solutions iterations in MEuro:",solutions)
print("Gap iterations in percentage:",gap_outputit)
print("Time iterations formulating in minutes:",time_formulating)
print("Time iterations solving in minutes:",time_solving)

import xlsxwriter 
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
workbook = xlsxwriter.Workbook(dt_string+'.xlsx')

worksheet0 = workbook.add_worksheet("Objective function values")  
worksheet0.write(0,0,"From the main driver")
worksheet0.write(0,1,-cost)
worksheet0.write(1,0,"From heuristics using all cables")
worksheet0.write(1,1,ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, elnet_layout2))
worksheet0.write(2,0,"From global opt. using all cables")
worksheet0.write(2,1,ee_dtu.calculate_irr(rated_rpm_array, Drotor_vector, Power_rated_array,hub_height_vector, water_depth_array, 
                    aep_final, Cost_collection))
worksheet0.write(3,0,"AEP [GWh]")
worksheet0.write(3,1,sum(aep_final)/(10**6))
worksheet0.write(4,0,"Time main driver [hours]")
worksheet0.write(4,1,elapsed1/3600)
worksheet0.write(5,0,"iterations main driver")
worksheet0.write(5,1,iterations)
worksheet0.write(6,0,"Cables included?")
worksheet0.write(6,1,optimized_with_cables)
worksheet0.write(7,0,"Centroid OSS?")
worksheet0.write(7,1,centroid_oss)
worksheet0.write(8,0,"Correction OSS?")
worksheet0.write(8,1,Correction)

worksheet1 = workbook.add_worksheet("WTs positions")  
worksheet1.write(0,0,"Abscissa [m]")
worksheet1.write(0,1,"Ordinate [m]")
row = 1
for i in range(len(X)):
    worksheet1.write(row,0,X[i])
    worksheet1.write(row,1,Y[i])
    row+=1
worksheet2 = workbook.add_worksheet("Global optimizer cable design")  
worksheet2.write(0,0,"Node 1")
worksheet2.write(0,1,"Node 2")
worksheet2.write(0,2,"length [m]")
worksheet2.write(0,3,"Cable type")
worksheet2.write(0,4,"Number of WTs downstream")
worksheet2.write(0,5,"Connection cost [Euros]")
row = 1
for i in range(Tree.shape[0]):
    worksheet2.write(row,0,Tree[i,0])
    worksheet2.write(row,1,Tree[i,1])
    worksheet2.write(row,2,Tree[i,2])
    worksheet2.write(row,3,Tree[i,3])
    worksheet2.write(row,4,Tree[i,4])
    worksheet2.write(row,5,Tree[i,5])
    row+=1
worksheet3 = workbook.add_worksheet("Heur. optimizer cable design")  
worksheet3.write(0,0,"Node 1")
worksheet3.write(0,1,"Node 2")
worksheet3.write(0,2,"length [m]")
worksheet3.write(0,3,"Cable type")
worksheet3.write(0,4,"Connection cost [Euros]")
row = 1
for i in range(T2.shape[0]):
    worksheet3.write(row,0,T2[i,0])
    worksheet3.write(row,1,T2[i,1])
    worksheet3.write(row,2,T2[i,2])
    worksheet3.write(row,3,T2[i,3])
    worksheet3.write(row,4,T2[i,4])
    row+=1
worksheet4 = workbook.add_worksheet("Global iterations")  
worksheet4.write(0,0,"Solutions value [MEuro]")
worksheet4.write(0,1,"Gap [%]")
worksheet4.write(0,2,"Time formulating [min]")
worksheet4.write(0,3,"Time solving [min]")
row = 1
for i in range(len(solutions)):
    worksheet4.write(row,0,solutions[i])
    worksheet4.write(row,1,gap_outputit[i])
    worksheet4.write(row,2,time_formulating[i])
    worksheet4.write(row,3,time_solving[i])
    row+=1
    
worksheet5 = workbook.add_worksheet("Main driver cost evolution")  
worksheet5.write(0,0,"Objective function")
row = 1
for i in range(len(recorder.driver_iteration_dict['cost'])):
    worksheet5.write(row,0,-recorder.driver_iteration_dict['cost'][i][0])
    row+=1
    
workbook.close() 

#plt.close()
#print(i)
#cost_vectorized[i]=cost
#locs[i,:,0]=state['x']
#locs[i,:,1]=state['y']

#plt.plot(-cost_vectorized)
#t = time.time()
#T2,elnet_layout2 = collection_system(X,Y,option,6,False,max_it,CablesForGlobal,False)
#elapsed = time.time() - t
#print(elapsed)
