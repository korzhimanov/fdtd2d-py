# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

import numpy as np
import h5py
import fdtd2d

import pytest

def laser_pulse_gauss(x_min, width, duration):
    return lambda t, x, y: np.exp(-(y/width)**2)*np.exp(
            -((t - (x - x_min))/duration-2.0)**2)*np.sin(2.*np.pi*(t - x))

@pytest.fixture
def params_correct():
    params = {}
    params['x_bounds'] = (0.0, 8.0)
    params['y_bounds'] = (-4.0, 4.0)
    params['matrix_size'] = {'x':64, 'y':64}
    params['time_steps'] = 80
    params['laser_pulse_y_shape'] = laser_pulse_gauss(params['x_bounds'][0], 1., 2.)
    params['laser_pulse_z_shape'] = laser_pulse_gauss(params['x_bounds'][0], 1., 2.)
    return params

@pytest.fixture
def params_calculated(params_correct):
    return fdtd2d.calculate_params(params_correct)

@pytest.fixture
def data_empty(params_calculated):
    nx = params_calculated['matrix_size']['x']
    ny = params_calculated['matrix_size']['y']
    data = {}
    for name in ('ex','ey','ezx','ezy','ez','hx','hy','hzx','hzy','hz'):
        data[name] = np.zeros(shape=(nx,ny), dtype=np.float64)
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

def test_field_generator_x_min(data_empty, params_calculated):
    d = copy.deepcopy(data_empty)
    p = params_calculated
    fdtd2d.generate_fields_x_min(d, 0.0, p)
    #np.savez_compressed('data_test/field_generator_x_min',
    #                    ey=d['ey'], hzx=d['hzx'], hz=d['hz'],
    #                    hy=d['hy'], ezx=d['ezx'], ez=d['ez'])
    with np.load('data_test/field_generator_x_min.npz') as correct_data:
        for name in ('ey','hzx','hz','hy','ezx','ez'):
            assert((d[name] == correct_data[name]).all())
    
def test_hdf5(params_correct):
    fdtd2d.run(params_correct)
    assert(os.path.exists("output.hdf5"))
    with h5py.File("output.hdf5", "r") as f:
        assert("ey" in f)

def test_run(params_correct):
    fdtd2d.run(params_correct)
    assert(os.path.exists("output.hdf5"))
    with h5py.File("output.hdf5", "r") as f:
        assert(f["ez"] == np.zeros(shape=(64,64), dtype=np.float64))
