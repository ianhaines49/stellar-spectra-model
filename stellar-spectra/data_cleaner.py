import numpy as np
from astropy.io import fits
import tqdm

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

def check_bitmask(mask, bad_bits):
    return mask & bad_bits


def data_extractor(fits_path = 'fits_files',
                   allStar_path = 'allStar-r12-l33.fits',
                   cluster_list = ['060+00', 'M15', 'N6791', 'K2_C4_168-21']):
    '''
    Extracts relevant data from .fits files for spectra model.
    
    Takes .fits file and for each star checks first if it is in one of the
    stellar clusters in name_list. If the star is, the program checks if
    the stellar parameters are in the correct range. If they are, then the
    program extracts the spectrum, errors, bitmask, .fits filename for star,
    signal-to-noise ratio, effective temperature, log of surface gravity,
    [Fe/H], [Mg/Fe], [Si/Fe], and any miscellaneous information. Information is
    organized in arrays, and then organized as a dictionary. Each dictionary is
    then added to a final array.
    
    Parameters
    ----------
    fits_path = string
        local path to the fits files grabbed from SDSS data archive
    allStar_path = string
        local path to the allStar data file
    name_list = list
        list of names of stellar clusters to extract information for
        
    Returns
    -------
    list
        list of dicts where each dict represents data for one star '''

    allStar = fits.open(allStar_path)

    corner_teff = []
    corner_logg = []
    corner_fe_h = []
    corner_mg_fe = []
    corner_si_fe = []
    star_data_arr = []

    for i in tqdm.tqdm(range(len(allStar[1].data))):
        
        star_data = {}
        
        if allStar[1].data[i][7] in cluster_list:
            # checks T_eff, log(g), [Fe/H], [Mg/Fe], [Si/Fe], and SNR
            if 0 < allStar[1].data[i][75] < 5700 and \
                0 < allStar[1].data[i][77] < 4 and \
                -1 < allStar[1].data[i][114] and \
                -9998 < allStar[1].data[i][102] and \
                -9998 < allStar[1].data[i][104] and \
                allStar[1].data[i][31] > 50:
                
                # opens .fits file for specific star
                hdul = fits.open(f'{fits_path}/apStar-r12-{allStar[1].data[i][4]}.fits')
                
                if isinstance(hdul[1].data[0], np.ndarray):
                    corner_teff.append(allStar[1].data[i][75])
                    corner_logg.append(allStar[1].data[i][77])
                    corner_fe_h.append(allStar[1].data[i][114])
                    corner_mg_fe.append(allStar[1].data[i][102])
                    corner_si_fe.append(allStar[1].data[i][104])

                    star_data['spectrum'] = hdul[1].data[0]
                    star_data['errors'] = hdul[2].data[0]
                    star_data['bitmask'] = hdul[3].data[0]
                    star_data['filename'] = allStar[1].data[i][4]
                    star_data['snr'] = allStar[1].data[i][31]
                    star_data['teff'] = allStar[1].data[i][75]
                    star_data['logg'] = allStar[1].data[i][77]
                    star_data['fe_h'] = allStar[1].data[i][114]
                    star_data['mg_fe'] = allStar[1].data[i][102]
                    star_data['si_fe'] = allStar[1].data[i][104]
                    star_data['allStar_misc'] = allStar[1].data[i]
                    star_data_arr.append(star_data)

    return star_data_arr

