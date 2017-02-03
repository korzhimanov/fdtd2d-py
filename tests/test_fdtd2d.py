# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import numpy as np
import h5py
import numba

import fdtd2d

import pytest

def laser_pulse_gauss(x_min, width, duration):
    @numba.jit(nopython=True)
    def f(t, x, y):
        return np.exp(-(y/width)**2)*np.exp(
            -((t - (x - x_min))/duration-2.0)**2)*np.sin(2.*np.pi*(t - x))
    return f
    #return lambda t, x, y: np.exp(-(y/width)**2)*np.exp(
    #        -((t - (x - x_min))/duration-2.0)**2)*np.sin(2.*np.pi*(t - x))

@pytest.fixture
def params_correct():
    params = {}
    params['x_bounds'] = (0.0, 8.0)
    params['y_bounds'] = (-4.0, 4.0)
    params['matrix_size'] = {'x':64, 'y':64}
    params['time_steps'] = np.int32(81)
    params['laser_pulse_y_shape'] = laser_pulse_gauss(params['x_bounds'][0], 1., 2.)
    params['laser_pulse_z_shape'] = laser_pulse_gauss(params['x_bounds'][0], 1., 2.)
    params["output"] = {}
    params["output"]["iteration_pass"] = 10
    params["output"]["directory_name"] = "tests/output"
    return params

@pytest.fixture
def params_calculated(params_correct):
    return fdtd2d.calculate_params(params_correct)

@pytest.fixture
def data_empty(params_calculated):
    nx = params_calculated['matrix_size']['x']
    ny = params_calculated['matrix_size']['y']
    data = {}
    for name in ('ex','ey','ezx','ezy','ez'):
        data[name] = np.zeros(shape=(nx,ny), dtype=np.float64)
    for name in ('hx','hy','hzx','hzy','hz'):
        data[name] = np.zeros(shape=(nx-1,ny-1), dtype=np.float64)
    return data

def test_params_calculation(params_calculated):
    p = params_calculated
    assert(p['box_size']['x'] == 8.0)
    assert(p['box_size']['y'] == 8.0)
    assert(p['space_step']['x'] == 0.125)
    assert(p['space_step']['y'] == 0.125)
    assert(p['time_step'] == np.sqrt(2.0)/16.0)
    assert(p['cfl']['x'] == np.sqrt(0.5))
    assert(p['cfl']['y'] == np.sqrt(0.5))

def test_init_data(data_empty, params_calculated):
    d = fdtd2d.init_data(params_calculated)
    for name in ('ex','ey','ezx','ezy','ez','hx','hy','hzx','hzy','hz'):
        assert((d[name] == data_empty[name]).all())
    
def test_field_generator_x_min(data_empty, params_calculated):
    d = copy.deepcopy(data_empty)
    p = params_calculated
    fdtd2d.generate_fields_x_min(d, 0.0, p)
#    with h5py.File('tests/data/field_generator_x_min.hdf5','w') as f:
#        for key, value in d.items():
#            f.create_dataset(key, data=value)
    with h5py.File('tests/data/field_generator_x_min.hdf5','r') as correct_data:
        for key in correct_data.keys():
            assert((d[key] == correct_data[key]).all())

def test_make_step_with_empty_data(data_empty, params_calculated):
    d = copy.deepcopy(data_empty)
    fdtd2d.make_step(d, params_calculated)
    for name in ('ey','hzx','hz','hy','ezx','ez'):
        assert((d[name] == data_empty[name]).all())

def test_make_step(data_empty, params_calculated):
#    _data = copy.deepcopy(data_empty)
#    for k in range(params_calculated['time_steps']*2//3):
#        _time = k*params_calculated['time_step']
#        fdtd2d.generate_fields_x_min(_data, _time, params_calculated)
#        fdtd2d.make_step(_data, params_calculated)
#    with h5py.File('tests/data/make_step_init.hdf5','w') as f:
#        for key, value in _data.items():
#            f.create_dataset(key, data=value)
#    fdtd2d.make_step(_data, params_calculated)
#    with h5py.File('tests/data/make_step.hdf5','w') as f:
#        for key, value in _data.items():
#            f.create_dataset(key, data=value)
    with h5py.File('tests/data/make_step_init.hdf5','r') as f, \
         h5py.File('tests/data/make_step.hdf5','r') as correct_data:
        data = copy.deepcopy(data_empty)
        for key in f.keys():
            data[key] = np.array(f[key])
        fdtd2d.make_step(data, params_calculated)
        for name in ('ex','ey','ezx','ezy','ez','hx','hy','hzx','hzy','hz'):
            assert(np.sum(data[name] - correct_data[name]) == 0.0)

def test_save_to_hdf5(data_empty):
    fdtd2d.save_to_hdf5("tests/output", 0, data_empty)
    
    assert(os.path.exists("tests/output/data_0.hdf5"))
    
    loaded_data = {}
    with h5py.File("tests/output/data_0.hdf5", "r") as f:
        for key, value in data_empty.items():
            assert(key in f)
            loaded_data[key] = np.array(f[key])
            assert(np.sum(loaded_data[key] - data_empty[key]) == 0.0)

def test_output(data_empty, params_calculated):
    assert(not fdtd2d.output(data_empty, 1, params_calculated))
    
    assert(fdtd2d.output(data_empty, 10, params_calculated))
    assert(os.path.exists("tests/output/data_1.hdf5"))
    
    loaded_data = {}
    with h5py.File("tests/output/data_1.hdf5", "r") as f:
        for key, value in data_empty.items():
            assert(key in f)
            loaded_data[key] = np.array(f[key])
            assert(np.sum(loaded_data[key] - data_empty[key]) == 0.0)

def test_run(params_correct):
    fdtd2d.run(params_correct)
    for n in range(9):
        assert(os.path.exists("tests/output/data_{}.hdf5".format(n)))
    with h5py.File("tests/output/data_8.hdf5", "r") as calculated_data, \
         h5py.File("tests/data/run.hdf5", "r") as correct_data:
        for key in correct_data.keys():
            assert(np.sum(np.array(calculated_data[key])
                        - np.array(correct_data[key])) == 0.0)
