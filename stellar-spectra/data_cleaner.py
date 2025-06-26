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

    # 4351 = 0b100001111111 corresponds to all bad pixels
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
    return wavelengths[continuum_pixels['is_continuum'] == True]
    
def closest_value(cont_wavelengths, spec_wavelengths):
    '''
    Returns spectra wavelengths closest to continuum wavelengths.
    
    Function performs a binary search to find spectra wavelengths
    closest to the continuum wavelengths.
    
    Parameters
    ----------
    cont_wavelengths : numpy array
        array containing all of the continuum wavelengths as determined by
        successive application of The Cannon model
    spec_wavelengths : numpy array
        array containing the wavelengths of a given spectrum
        
    Returns
    -------
    fit_wavelengths : numpy array
        array containing the spectrum wavelengths closest to continuum, ready
        to be used for fitting spectra
    '''

    n, m = cont_wavelengths.size, spec_wavelengths.size

    fit_wavelengths = np.empty(n, dtype=cont_wavelengths.dtype)

    j = 0    # index pointer for spec_wavelengths
    for i in range(n):
        # steps through spec_wavelengths until it finds val closest to continuum
        while j + 1 < m and abs(spec_wavelengths[j+1] - cont_wavelengths[i]) < \
                            abs(spec_wavelengths[j] - cont_wavelengths[i]):
            j += 1
        fit_wavelengths[i] = spec_wavelengths[j]

    return fit_wavelengths
