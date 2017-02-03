# -*- coding: utf-8 -*-

import math
import copy

def calculate_params(init_params):
    """Calculate parameters needed for simulation and return them as a dictionary."""
    params = copy.deepcopy(init_params)
    
    params['box_size'] = {}
    params['box_size']['x'] = params['x_bounds'][1] - params['x_bounds'][0]
    params['box_size']['y'] = params['y_bounds'][1] - params['y_bounds'][0]
    
    params['space_step'] = {}
    params['space_step']['x'] = params['box_size']['x']/params['matrix_size']['x']
    params['space_step']['y'] = params['box_size']['y']/params['matrix_size']['y']
    
    params['time_step'] = (
            math.sqrt(params['space_step']['x']**2 +
                      params['space_step']['y']**2))/2.0
    
    params['cfl'] = {}
    params['cfl']['x'] = params['time_step']/params['space_step']['x']
    params['cfl']['y'] = params['time_step']/params['space_step']['y']
    
    return params

def generate_fields_x_min(data, time, params):
    """Generate fields at left boundary (x_min = params['x_bounds'][0]) at time moment 'time'"""
    for j in range(params['matrix_size']['y']):
        data['ey'] [1,j] -= params['laser_pulse_y_shape'](
                time,
                params['x_bounds'][0] + params['space_step']['x'],
                params['y_bounds'][0] + j*params['space_step']['y'])
        data['ezx'][1,j] -= params['laser_pulse_z_shape'](
                time + 0.5*params['time_step'],
                params['x_bounds'][0] + 1.5*params['space_step']['x'],
                params['y_bounds'][0] + (j-0.5)*params['space_step']['y'])
        data['ez'] [1,j]  = data['ezx'][1,j] + data['ezy'][1,j]
    
    for j in range(params['matrix_size']['y']-1):
        data['hy'] [1,j] -= params['laser_pulse_z_shape'](
                time,
                params['x_bounds'][0] + params['space_step']['x'],
                params['y_bounds'][0] + j*params['space_step']['y'])
        data['hzx'][1,j] -= params['laser_pulse_y_shape'](
                time + 0.5*params['time_step'],
                params['x_bounds'][0] + 1.5*params['space_step']['x'],
                params['y_bounds'][0] + (j-0.5)*params['space_step']['y'])
        data['hz'] [1,j]  = data['hzx'][1,j] + data['hzy'][1,j]

def run(init_params_dict):
    """Run the simulations..."""
    params_dict = calculate_params(init_params_dict)
    print(params_dict)
    print('Running...')