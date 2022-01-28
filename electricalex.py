#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 07:41:41 2020

@author: katdyk
"""

import numpy as np
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm import TopFarmGroup, TopFarmProblem
from topfarm.easy_drivers import EasyRandomSearchDriver, EasyScipyOptimizeDriver
from topfarm.drivers.random_search_driver import RandomizeTurbinePosition_Circle
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint
from topfarm.cost_models.electrical.simple_msp import ElNetLength, ElNetCost 
import matplotlib.pylab as plt

# Convenient plot component for plotting the cables from simple_msp in jupyter notebooks
from workshop.cabling import XYCablePlotComp

from workshop.cabling import get_site
from py_wake.examples.data.DTU10MW_RWT import DTU10MW



site = get_site()
n_wt = len(site.initial_position)
windTurbines = DTU10MW()
Drotor_vector = [windTurbines.diameter()] * n_wt 
power_rated_vector = [float(windTurbines.power(20)/1000)] * n_wt 
hub_height_vector = [windTurbines.hub_height()] * n_wt 
rated_rpm_array = 12. * np.ones([n_wt])

print('Number of turbines:', n_wt)

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

from topfarm.cost_models.economic_models.turbine_cost import economic_evaluation as EE_NREL

def irr_nrel(aep, electrical_connection_cost, **kwargs):
    return EE_NREL(Drotor_vector, power_rated_vector, hub_height_vector, aep, electrical_connection_cost).calculate_irr()


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

## Some user options
#@markdown Which IRR Cost model to use
IRR_COST = 'DTU' #@param ["DTU", "NREL"]

#@markdown Minimum spacing between the turbines
min_spacing = 2 #@param {type:"slider", min:2, max:10, step:1}

#@markdown Minimum spacing between the turbines
cable_cost_per_meter = 750. #@param {type:"slider", min:0, max:10000, step:1}

## Electrical grid cable components (Minimum spanning tree from Topfarm report 2010)
elnetlength = ElNetLength(n_wt=n_wt)
elnetcost = ElNetCost(n_wt=n_wt, output_key='electrical_connection_cost', cost_per_meter=cable_cost_per_meter)

# The Topfarm IRR cost model components
irr_dtu_comp = CostModelComponent(input_keys=['aep', ('electrical_connection_cost', 0.0)],   n_wt=n_wt, 
                                  cost_function=irr_dtu, output_key="irr", output_unit="%",   objective=True, income_model=True)
irr_nrel_comp = CostModelComponent(input_keys=['aep', ('electrical_connection_cost', 0.0)],   n_wt=n_wt, 
                                   cost_function=irr_nrel, output_key="irr", output_unit="%",   objective=True, income_model=True)
irr_cost_models = {'DTU': irr_dtu_comp, 'NREL': irr_nrel_comp}


## The Topfarm AEP component
aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt, cost_function=aep_func, 
                              output_key="aep", output_unit="GWh", objective=False, output_val=np.zeros(n_wt))

## Plotting component
plot_comp = XYCablePlotComp(memory=0, plot_improvements_only=False, plot_initial=False)


## The group containing all the components
group = TopFarmGroup([aep_comp, elnetlength, elnetcost, irr_cost_models[IRR_COST]])

problem = TopFarmProblem(
        design_vars=dict(zip('xy', site.initial_position.T)),
        cost_comp=group,
        driver=EasyRandomSearchDriver(randomize_func=RandomizeTurbinePosition_Circle(), max_iter=10),
        #driver=EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=20, tol=1e-3, disp=False),
        constraints=[SpacingConstraint(min_spacing * windTurbines.diameter(0)),
                     XYBoundaryConstraint(site.boundary)],
        expected_cost=1.0,
        plot_comp=plot_comp)

from openmdao.api import pyOptSparseDriver
problem.driver = pyOptSparseDriver(optimizer='ALPSO')
problem.driver.opt_settings['maxOuterIter'] = 20

cost, state, recorder = problem.optimize()


