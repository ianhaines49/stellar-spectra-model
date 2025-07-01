import numpy as np
from astropy.io import fits

def get_continuum_wavelengths(filepath):
    '''
    Determines which wavelengths are closest to continuum.

    Function takes in supplied continuum pixels filename and spectrum pixels
    file, and returns an array of wavelengths closest to the continuum pixels.

    Parameters
    ----------
    filename : string
        contains a dict containing an array of pixels and an array of booleans
        saying which pixels are continuum pixels
    
    Returns
    -------
    out : ndarray
        array containing continuum wavelengths
    '''

    continuum_pixels = np.load(f'{filepath}')
    wavelengths = continuum_pixels['wavelength']
    return wavelengths[continuum_pixels['is_continuum'] == True]
    
def closest_value(cont_wavelengths, spec_wavelengths):
    '''
    Returns wavelength indices closest to continuum wavelengths.
    
    Function performs a binary search to find spectra wavelengths
    closest to the continuum wavelengths and returns the indices of
    closest values.
    
    Parameters
    ----------
    cont_wavelengths : ndarray
        array containing all of the continuum wavelengths as determined by
        successive application of The Cannon model
    spec_wavelengths : ndarray
        array containing the wavelengths of a given spectrum
        
    Returns
    -------
    fit_bools : ndarray
        array containing the spectrum wavelengths closest to continuum, ready
        to be used for fitting spectra
    '''

    n, m = cont_wavelengths.size, spec_wavelengths.size
    fit_bools = np.full(m, False)
    j = 0    # Index pointer for spec_wavelengths
    for i in range(n):
        # Steps through spec_wavelengths until it finds val closest to continuum
        while j + 1 < m and abs(spec_wavelengths[j+1] - cont_wavelengths[i]) \
                            < abs(spec_wavelengths[j] - cont_wavelengths[i]):
            j += 1
        fit_bools[j] = True

    return fit_bools

def bad_pix_indices(bitmask_array, bad_pixels_dict, len):
    '''
    Gets the indices of bad pixels.
    
    Function takes in the errors and bitmask arrays associated with each
    .fits file and updates the errors based on the bitmask. The structure of
    the APOGEE pixel error flags and further info can be found at
    https://www.sdss4.org/dr17/algorithms/bitmasks/#APOGEE_PIXMASK

    Parameters
    ----------
    bitmask_array : ndrray
        array of bitmasks indicating what errors each pixel has
    bad_pixels_dict : dictionary
        dictionary of which pixel errors are significant
    
    Returns
    -------
    bad_pixels : ndarray
        indices of bad pixels to be ignored when performing WLS
    '''

    # Creates bitmask for all bad pixels provided in bad_pixels_dict
    bit_sum = 0
    for key in bad_pixels_dict:
        bit_sum += 2**int(key)

    bad_bitmask = np.full(len, bit_sum)
    
    return np.where(bitmask_array & bad_bitmask > 0)[0]

def update_errors(data, bad_pixels_dict):
    '''
    Updates errors based on bitmasks
    
    Function takes apStar HDUList data and a dictionary of which pixels are
    bad and creates an updated errors array where bad pixels have an error
    of 1E+10, the default value for unused pixels.
    
    Parameters
    ----------
    data : HDUList
        apStar data; data[2] is error spectra; data[3] is mask spectra
    bad_pixels_dict : dictionary
        dictionary format must be {'<digit from 0-14>' : 1}; if a digit is
        not included as a key it is assumed that there is no pixel error    
    '''
    updated_errors = data[2].data.copy()
    mask_spectra = data[3].data.copy()
    nwave = data[0].header['nwave']
    if type(mask_spectra[0]) == np.ndarray:
        updated_errors[0][bad_pix_indices(mask_spectra[0], bad_pixels_dict, nwave)] = 1E+10
        return updated_errors[0]
    
    updated_errors[bad_pix_indices(mask_spectra, bad_pixels_dict, nwave)] = 1E+10
    return updated_errors

def create_wavelengths(data):
    '''Creates apStar wavelengths from HDUList.

    Creates apStar wavelengths from HDUList by accessing header values.
    Returns linear wavelength array values.

    Parameters
    ----------
    data : HDUList
        apStar HDUList file data

    Returns
    -------
    out : ndarray
        linearly spaced array of wavelengths
    '''

    # Initialize spectrum and continuum wavelengths data
    start = data[1].header['crval1']
    delta = data[1].header['cdelt1']
    return np.logspace(start, start + delta*8575, 8575)

def interval_cut(data, interval):
    '''Returns data within a specific interval.'''

    start = interval[0]
    end = interval[1]
    return np.intersect1d(np.where(start < data)[0], np.where(data < end)[0])

def data_normalizer(data, filepath, interval, bad_pix_dict):
    '''
    Takes in apStar data and returns a normalized version.
    
    Function normalizes data by fitting with Chebyshev polynomials, then
    evaluates Chebyshev series for each point to determine normalized value.
    
    Parameters
    ----------
    data : HDUList
        expected to be apStar HDU file
    filepath : string
        path to continuum_pixels_apogee.npz file
    interval : arraylike or tuple
        endpoints of the wavelength range
    Returns
    -------
    out : ndarray
        normalized apStar file data
    '''

    # Initialize spectrum and continuum wavelengths data
    spec_wavelengths = create_wavelengths(data)
    cont_wavelengths = get_continuum_wavelengths(filepath)
    spectrum = data[1].data[0].copy()
    errors = update_errors(data, bad_pix_dict)

    # Determine indices of data in correct interval
    cut_inds = interval_cut(spec_wavelengths, interval)
    cont_cut_inds = interval_cut(cont_wavelengths, interval)

    # Cut spectra data
    spec_wavelengths = spec_wavelengths[cut_inds]
    cont_wavelengths = cont_wavelengths[cont_cut_inds]
    spectrum = spectrum[cut_inds]
    errors = errors[cut_inds]
    
    # Get continuum spectrum wavelengths, spectrum, and errors
    cont_inds = closest_value(cont_wavelengths, spec_wavelengths)
    ctm_spec_wavelengths = spec_wavelengths[cont_inds]
    ctm_spectrum = spectrum[cont_inds]
    ctm_errors = errors[cont_inds]

    # Fit with Cheby polys, and calculate normalization
    cheby_coeffs = np.polynomial.chebyshev.Chebyshev.fit(
        x=ctm_spec_wavelengths, y=ctm_spectrum, deg=2,
        window=[spec_wavelengths[0],spec_wavelengths[-1]],
        w=1/ctm_errors
        )

    # Calculate normalization for interval
    dense_cut_inds = interval_cut(spec_wavelengths, interval)
    dense_spec_wavelengths = spec_wavelengths[dense_cut_inds]
    dense_spectrum = spectrum[dense_cut_inds]
    dense_errors = errors[dense_cut_inds]
    normalization = np.polynomial.chebyshev.chebval(dense_spec_wavelengths,
                                                    cheby_coeffs.coef)

    return [dense_spectrum/normalization, dense_errors/normalization]