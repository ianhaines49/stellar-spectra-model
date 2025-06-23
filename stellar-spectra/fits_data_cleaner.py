import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from astropy.io import fits
import corner
from matplotlib import rcParams
import os
import math
from scipy import optimize
from scipy import stats
import time
import random
import tqdm
import pymc3 as pm
import arviz as az
import theano
import theano.tensor as tt
from arviz import plot_trace as traceplot

def subtract(n):
    '''Function that determines which digit is in a bitmask.

    Function is meant to be called recursively to determine all of the digits in a
    binary representation until the remainder is zero.
    Input:
    n: input digit
    Output:
    list: a list of a new digit and a flag telling which digit was in the sum.
    '''
    
    if n-1 <= 0: 
        return [n-1, 0] 
    elif n-2 <= 0:     
        return [n-1, 1]
    elif n-4 <= 0:    
        return [n-2, 2]
    elif n-8 <= 0:
        return [n-4, 3]
    elif n-16 <= 0:    
        return [n-8, 4]
    elif n-32 <= 0:    
        return [n-16, 5]
    elif n-64 <= 0:   
        return [n-32, 6]
    elif n-128 <= 0:    
        return [n-64, 7]
    elif n-256 <= 0:    
        if n-256 == 0:        
            return [0, 8]    
        else:        
            return [n-128, 8]
    elif n-512 <= 0:   
        if n-512 == 0:       
            return [0, 9]   
        else:       
            return [n-256, 9]
    elif n-1024 <= 0:    
        if n-1024 == 0:
            return [0, 10]
        else:
            return [n-512, 10]
    elif n-2048 <= 0:
        if n-2048 == 0:
            return [0, 11]
        else:  
            return [n-1024, 11]
    elif n-4096 <= 0:    
        return [n-2048, 12]
    elif n-8192 <= 0:    
        if n-8192 == 0:        
            return [0, 13]   
        else:   
            return [n-4096, 13]
    elif n-16384 <= 0:   
        if n-16384 == 0:       
            return [0, 14]   
        else:   
            return [n-8192, 14]
    else:   
        return [n-16384, 15]
    
def pixel_checker(error_arr, mask_arr):
    '''Function that adjusts errors based on bitmaskarray.
    
    Function that takes in an error array and a bitmask array to determine
    which errors need to be altered based upon the bitmasks.
    Function does so by iteratively calling subtract(n).
    Intput:
    error_arr: list of errors
    mask_arr: list of bitmasks
    Output:
    error_arr: updated list of new errors
    '''
    
    error_copy = [] 
    for val in error_arr:    
        error_copy.append(val)
        
    for i, n in zip(range(len(error_arr)), mask_arr):
        while n != 0:
            tup = subtract(n)
            n = tup[0]
            if tup[1] == 0 or tup[1] == 1 or tup[1] == 2 or tup[1] == 3 or\
               tup[1] == 4 or tup[1] == 5 or tup[1] == 6 or tup[1] == 7 or tup[1] == 12:
                error_copy[i] = 10**10
                break           
    return error_copy