# -*- coding: utf-8 -*-

import math

def calculate_params(init_params):
    """Calculate parameters needed for simulation and return them as a dictionary."""
    params = init_params.copy()
    
    params['box_size'] = {}
    params['box_size']['x'] = params['x_bounds'][1] - params['x_bounds'][0]
    params['box_size']['y'] = params['y_bounds'][1] - params['y_bounds'][0]
    
    params['space_step'] = {}
    params['space_step']['x'] = params['box_size']['x']/params['matrix_size']['x']
    params['space_step']['y'] = params['box_size']['y']/params['matrix_size']['y']
    
    params['time_step'] = (
            math.sqrt(params['space_step']['x']**2 +
                      params['space_step']['y']**2)
            )/2.0
    
    params['cfl'] = {}
    params['cfl']['x'] = params['time_step']/params['space_step']['x']
    params['cfl']['y'] = params['time_step']/params['space_step']['y']
    
    return params

def run(init_params_dict):
    """Run the simulations..."""
    params_dict = calculate_params(init_params_dict)
    print(params_dict)
    print('Running...')