# -*- coding: utf-8 -*-

import copy
import numpy as np
import h5py
import numba

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
    
    _params['y1'] = np.asarray(range(_params['matrix_size']['y']), dtype=np.float64) * \
                         _params['space_step']['y'] + _params['y_bounds'][0]
    _params['y2'] = np.asarray(range(_params['matrix_size']['y']), dtype=np.float64) * \
                          _params['space_step']['y'] - 0.5*_params['space_step']['y'] + \
                          _params['y_bounds'][0]
    return _params

def init_data(params):
    _data = {}
    for name in ('ex','ey','ezx','ezy','ez','hx','hy','hzx','hzy','hz'):
        _data[name] = np.zeros(
                shape=(params['matrix_size']['x'], params['matrix_size']['x']),
                dtype=np.float64)
    return _data

@numba.jit
def generate_ey_x_min(ey, shape, t, x, y):
    ey[1] -= shape(t, x, y)

@numba.jit
def generate_hzx_x_min(hzx, shape, t, x, y):
    hzx[1,:-1] -= shape(t, x, y[:-1])

@numba.jit
def generate_hy_x_min(hy, shape, t, x, y):
    hy[1,:-1] -= shape(t, x, y[:-1])

@numba.jit
def generate_ezx_x_min(ezx, shape, t, x, y):
    ezx[1] -= shape(t, x, y)

@numba.jit(nopython=True)
def generate_ez_x_min(ez, ezx, ezy):
    ez[1] = ezx[1] + ezy[1]

@numba.jit(nopython=True)
def generate_hz_x_min(hz, hzx, hzy):
    hz[1] = hzx[1] + hzy[1]

@numba.jit
def generate_fields_x_min(d, time, p):
    """Generate fields at left boundary (x_min = p['x_bounds'][0]) at time moment 'time'"""
    generate_ey_x_min(d['ey'],
                      p['laser_pulse_y_shape'],
                      time,
                      p['x_bounds'][0] + p['space_step']['x'],
                      p['y1'])
    generate_hzx_x_min(d['hzx'],
                      p['laser_pulse_y_shape'],
                      time + 0.5*p['time_step'],
                      p['x_bounds'][0] + 1.5*p['space_step']['x'],
                      p['y2'])
    
    generate_hy_x_min(d['hy'],
                      p['laser_pulse_z_shape'],
                      time + 0.5*p['time_step'],
                      p['x_bounds'][0] + 1.5*p['space_step']['x'],
                      p['y2'])
    generate_ezx_x_min(d['ezx'],
                      p['laser_pulse_z_shape'],
                      time,
                      p['x_bounds'][0] + p['space_step']['x'],
                      p['y1'])
    generate_ez_x_min(d['ez'], d['ezx'], d['ezy'])
    generate_hz_x_min(d['hz'], d['hzx'], d['hzy'])

@numba.jit
def update_ex(ex, hz, half_cfl):
#    ex[1:-1,1:-1] += half_cfl*(hz[1:-1,1:-1] + hz[:-2,1:-1] - hz[1:-1,:-2] - hz[:-2,:-2])
    for i in range(1,ex.shape[0]-1):
        for j in range(1,ex.shape[1]-1):
            ex[i,j] += half_cfl*(hz[i,j] + hz[i-1,j] - hz[i,j-1] - hz[i-1,j-1])

@numba.jit
def update_ey(ey, hz, half_cfl):
    for i in range(1,ey.shape[0]-1):
        for j in range(1,ey.shape[1]-1):
            ey[i,j] -= half_cfl*(hz[i,j] + hz[i,j-1] - hz[i-1,j] - hz[i-1,j-1])

@numba.jit
def update_ezx(ezx, hy, half_cfl):
    for i in range(1,ezx.shape[0]-1):
        for j in range(1,ezx.shape[1]-1):
            ezx[i,j] += half_cfl*(hy[i,j] + hy[i,j-1] - hy[i-1,j] - hy[i-1,j-1])

@numba.jit
def update_ezy(ezy, hx, half_cfl):
    for i in range(1,ezy.shape[0]-1):
        for j in range(1,ezy.shape[1]-1):
            ezy[i,j] -= half_cfl*(hx[i,j] + hx[i-1,j] - hx[i,j-1] - hx[i-1,j-1])

@numba.jit
def update_ez(ez, ezx, ezy):
    for i in range(1,ez.shape[0]-1):
        for j in range(1,ez.shape[1]-1):
            ez[i,j] = ezx[i,j] + ezy[i,j]

@numba.jit
def update_hx(hx, ez, half_cfl):
    for i in range(hx.shape[0]-1):
        for j in range(hx.shape[1]-1):
            hx[i,j] -= half_cfl*(ez[i+1,j+1] + ez[i,j+1] - ez[i+1,j] - ez[i,j])

@numba.jit
def update_hy(hy, ez, half_cfl):
    for i in range(hy.shape[0]-1):
        for j in range(hy.shape[1]-1):
            hy[i,j] += half_cfl*(ez[i+1,j+1] + ez[i+1,j] - ez[i,j+1] - ez[i,j])

@numba.jit
def update_hzx(hzx, ey, half_cfl):
    for i in range(hzx.shape[0]-1):
        for j in range(hzx.shape[1]-1):
            hzx[i,j] -= half_cfl*(ey[i+1,j+1] + ey[i+1,j] - ey[i,j+1] - ey[i,j])

@numba.jit
def update_hzy(hzy, ex, half_cfl):
    for i in range(hzy.shape[0]-1):
        for j in range(hzy.shape[1]-1):
            hzy[i,j] += half_cfl*(ex[i+1,j+1] + ex[i,j+1] - ex[i+1,j] - ex[i,j])

@numba.jit
def update_hz(hz, hzx, hzy):
    for i in range(hz.shape[0]-1):
        for j in range(hz.shape[1]-1):
            hz[i,j] = hzx[i,j] + hzy[i,j]

@numba.jit
def make_step(d, p):
    """Make step"""
    update_ex(d['ex'], d['hz'], 0.5*p['cfl']['y'])
    update_ey(d['ey'], d['hz'], 0.5*p['cfl']['x'])
    update_ezx(d['ezx'], d['hy'], 0.5*p['cfl']['x'])
    update_ezy(d['ezy'], d['hx'], 0.5*p['cfl']['y'])
    update_ez(d['ez'], d['ezx'], d['ezy'])
    
    update_hx(d['hx'], d['ez'], 0.5*p['cfl']['y'])
    update_hy(d['hy'], d['ez'], 0.5*p['cfl']['x'])
    update_hzx(d['hzx'], d['ey'], 0.5*p['cfl']['x'])
    update_hzy(d['hzy'], d['ex'], 0.5*p['cfl']['y'])
    update_hz(d['hz'], d['hzx'], d['hzy'])

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