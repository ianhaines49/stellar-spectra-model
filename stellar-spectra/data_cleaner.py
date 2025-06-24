import bisect
import numpy as np

def update_errors(error_array, bitmask_array):
    '''
    Updates the errors of spectrum at each pixel based on bitmask data.
    
    Function takes in the errors and bitmask arrays associated with each
    .fits file and updates the errors based on the bitmask. Errors are updated
    to be 10^10 so that the pixel doesn't affect the WLS fit. The structure of
    the APOGEE pixel error flags and further info can be found at
    https://www.sdss4.org/dr17/algorithms/bitmasks/#APOGEE_PIXMASK

    Parameters
    ----------
    error_array : numpy array
        array of the errors for each pixel of a given spectrum
    bitmask_array : numpy array
        array of bitmasks indicating which pixels are bad
        and what errors they have
    
    Returns
    -------
    numpy array
        updated errors to be used for WLS fitting
    '''

    #4351 = 0b100001111111 corresponds to all bad pixels
    error_array[bitmask_array & 4351 > 0] = 10**10
    
    return error_array

def get_continuum_wavelengths(cont_pix_filename):
    '''
    Determines which wavelengths are closest to continuum.

    Function takes in supplied continuum pixels filename and spectrum pixels
    file, and returns an array of wavelengths closest to the continuum pixels.

    Parameters
    ----------
    cont_pixels_filename : string
        contains a dict containing an array of pixels and an array of booleans
        saying which pixels are continuum pixels
    
    Returns
    -------
    numpy array
        array containing continuum wavelengths
    '''

    continuum_pixels = np.load(f'{cont_pix_filename}.npz')
    wavelengths = continuum_pixels['wavelength']
    cont_wavelengths = wavelengths[continuum_pixels['is_continuum'] == True]
    
