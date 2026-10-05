# -*- coding: utf-8 -*-
"""
Routines to apply conversion to given parameter lists
rotates hyperparameter cube to new user-defined rotated space
"""

import numpy as np

def rotate(th):
    return [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]

def inv_rotate(th):
    return np.linalg.inv(rotate(th))


def demo_rotation(X, coord_names, **kwargs):
    #coord_names will be low_level_coord_names from CEP
    print("dat received[0]:\n",X[0])
    
    #example kwargs usage: 
    if 'theta' in kwargs:
        theta = np.float(kwargs['theta'])
    else:
        theta = np.pi/4.0 #45 deg rotation
    
    #get gammas' indices in X_out(= X) from coord names
    x_rot = np.zeros((len(X),2))
    rot_cols = []
    for i in np.arange(2):
        #do one coord at a time
        indx = coord_names.index("x"+str(i)) 
        rot_cols.append(indx)
        x_rot[:,i] = X[:,indx]

    #apply transform
    x_prime = np.matmul(rotate(theta),x_rot.T).T
    X_out = X
    X_out[:,rot_cols] = x_prime #fill rotated values 
    return X_out


def inverse_demo_rotation(X, coord_names, **kwargs):
    print("dat received[0]:\n",X[0])
    
    if 'theta' in kwargs:
        theta = np.float(kwargs['theta'])
    else:
        theta = np.pi/4.0 #45 deg rotation
        
    x_prime_out = np.zeros((len(X),2))
    rot_cols = []
    for i in np.arange(2):
        #do one coord at a time
        indx = coord_names.index("x"+str(i)) #coord_names = dat_orig_names
        rot_cols.append(indx)
    
    #apply inverse
    x_prime_out = X[:,rot_cols]
    x_rot_post = np.matmul(inv_rotate(theta),x_prime_out.T).T
    X_out = X

    for i, col in enumerate(rot_cols): 
        X_out[:,col] = x_rot_post[:,i]
    return X_out


def get_bounds(param_list, bounds_dict, **kwargs):
    if 'buffer' in kwargs:
        buffer = np.float(kwargs['buffer'])
    else:
        buffer = 0.0
        print(" Warning: no buffer provided")
    
    rot_mat = (np.sqrt(2)/2)*np.array([[1,-1],[1,1]]) #45 degree rotation in 2D
    
    #set rotated bounds to use
    rot_coords = {}
    for key in ["x0","x1"]:
        #N.B.: requires bounds_dict to be defined: set in PUFF using --parameter-range
        x = np.array(bounds_dict[key])
        #rotate
        x_rot = np.matmul(rot_mat,x.T).T
        
        #apply hypercube buffer
        ubound = x_rot[1] + buffer*abs(x_rot[1])
        lbound = x_rot[0] - buffer*abs(x_rot[0])
        rot_coords[key] = [lbound,ubound]
    
    #put updated bounds into new dict (hopefully same order)
    buff_dict = {}
    i = 0
    for p in param_list:
        if p == "x"+str(i):
            buff_dict[p] = rot_coords[p]
            i += 1
        else:
            if p in bounds_dict.keys():
                buff_dict[p] = bounds_dict[p]
            else:
                print("  external bounds: unbounded parameter "+p)
    if i == 0:
        print(" BOUND ERROR: could not match buffered bounds to original")
    return buff_dict


