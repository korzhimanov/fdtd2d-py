# -*- coding: utf-8 -*-

import copy
import numpy as np
import h5py

def calculate_params(init_params):
    """Calculate parameters needed for simulation and return them as a dictionary."""
    _params = copy.deepcopy(init_params)
    
    _params['box_size'] = {}
    _params['box_size']['x'] = _params['x_bounds'][1] - _params['x_bounds'][0]
    _params['box_size']['y'] = _params['y_bounds'][1] - _params['y_bounds'][0]
    
    _params['space_step'] = {}
    _params['space_step']['x'] = _params['box_size']['x']/_params['matrix_size']['x']
    _params['space_step']['y'] = _params['box_size']['y']/_params['matrix_size']['y']
    
    _params['time_step'] = (
            np.sqrt(_params['space_step']['x']**2 +
                    _params['space_step']['y']**2))/2.0
    
    _params['cfl'] = {}
    _params['cfl']['x'] = _params['time_step']/_params['space_step']['x']
    _params['cfl']['y'] = _params['time_step']/_params['space_step']['y']
    
    return _params

def init_data(params):
    _data = {}
    for name in ('ex','ey','ezx','ezy','ez','hx','hy','hzx','hzy','hz'):
        _data[name] = np.zeros(
                shape=(params['matrix_size']['x'], params['matrix_size']['x']),
                dtype=np.float64)
    return _data

def generate_fields_x_min(d, time, p):
    """Generate fields at left boundary (x_min = p['x_bounds'][0]) at time moment 'time'"""
    for j in range(p['matrix_size']['y']):
        d['ey'] [1,j] -= p['laser_pulse_y_shape'](
                time,
                p['x_bounds'][0] + p['space_step']['x'],
                p['y_bounds'][0] + j*p['space_step']['y'])
        d['ezx'][1,j] -= p['laser_pulse_z_shape'](
                time + 0.5*p['time_step'],
                p['x_bounds'][0] + 1.5*p['space_step']['x'],
                p['y_bounds'][0] + (j-0.5)*p['space_step']['y'])
        d['ez'] [1,j]  = d['ezx'][1,j] + d['ezy'][1,j]
    
    for j in range(p['matrix_size']['y']-1):
        d['hy'] [1,j] -= p['laser_pulse_z_shape'](
                time,
                p['x_bounds'][0] + p['space_step']['x'],
                p['y_bounds'][0] + j*p['space_step']['y'])
        d['hzx'][1,j] -= p['laser_pulse_y_shape'](
                time + 0.5*p['time_step'],
                p['x_bounds'][0] + 1.5*p['space_step']['x'],
                p['y_bounds'][0] + (j-0.5)*p['space_step']['y'])
        d['hz'] [1,j]  = d['hzx'][1,j] + d['hzy'][1,j]

def make_step(d, p):
    """Make step"""
    for j in range(1,p['matrix_size']['y']-1):
        for i in range(1,p['matrix_size']['x']-1):
            d['ex'] [i,j] += 0.5*p['cfl']['y']*(
                    d['hz'][i,j] + d['hz'][i-1,j]
                    - d['hz'][i,j-1] - d['hz'][i-1,j-1])
            d['ey'] [i,j] -= 0.5*p['cfl']['x']*(
                    d['hz'][i,j] + d['hz'][i,j-1]
                    - d['hz'][i-1,j] - d['hz'][i-1,j-1])
            d['ezx'][i,j] += 0.5*p['cfl']['x']*(
                    d['hy'][i,j] + d['hy'][i,j-1]
                    - d['hy'][i-1,j] - d['hy'][i-1,j-1])
            d['ezy'][i,j] -= 0.5*p['cfl']['y']*(
                    d['hx'][i,j] + d['hx'][i-1,j]
                    - d['hx'][i,j-1] - d['hx'][i-1,j-1])
            d['ez'] [i,j] = d['ezx'][i,j] + d['ezy'][i,j]
            
    for j in range(p['matrix_size']['y']-1):
        for i in range(p['matrix_size']['x']-1):
            d['hx'] [i,j] -= 0.5*p['cfl']['y']*(
                    d['ez'][i+1,j+1] + d['ez'][i,j+1]
                    - d['ez'][i+1,j] - d['ez'][i,j])
            d['hy'] [i,j] += 0.5*p['cfl']['x']*(
                    d['ez'][i+1,j+1] + d['ez'][i+1,j]
                    - d['ez'][i,j+1] - d['ez'][i,j])
            d['hzx'][i,j] -= 0.5*p['cfl']['x']*(
                    d['ey'][i+1,j+1] + d['ey'][i+1,j]
                    - d['ey'][i,j+1] - d['ey'][i,j])
            d['hzy'][i,j] += 0.5*p['cfl']['y']*(
                    d['ex'][i+1,j+1] + d['ex'][i,j+1]
                    - d['ex'][i+1,j] - d['ex'][i,j])
            d['hz'] [i,j] = d['hzx'][i,j] + d['hzy'][i,j]

def save_to_hdf5(output_directory_name, n, data):
    """Save current data to hdf5 file with sub-index 'n'"""
    file_name = '{}/data_{}.hdf5'.format(output_directory_name, n)
    with h5py.File(file_name,'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)

def output(d, k, p):
    """Make output"""
    if k%p["output"]["iteration_pass"] == 0:
        save_to_hdf5(
                p["output"]["directory_name"],
                k//p["output"]["iteration_pass"],
                d)
        return True
    else:
        return False

def run(init_params_dict):
    """Run the simulations"""
    _params_dict = calculate_params(init_params_dict)
    _data = init_data(_params_dict)
    for k in range(_params_dict['time_steps']):
        _time = k*_params_dict['time_step']
        generate_fields_x_min(_data, _time, _params_dict)
        make_step(_data, _params_dict)
        output(_data, k, _params_dict)