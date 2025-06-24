def decimal_to_binary(n):
    '''Takes in decimal number and outputs its binary representation'''

    binary_val = ''
    while n > 0:
        binary_val = (n & 1) + binary_val
        n >>= 1
    return binary_val

def pixel_checker(error_array, bitmask_array):
    '''
    Function that updates the errors of spectrum based on bitmask data.
    
    This function takes in the errors and bitmask arrays associated with each
    .fits file and updates the errors based on the bitmask. This is done so
    a WLS fit can be performed. The structure of the APOGEE Pipeline, ASPCAP,
    at https://www.sdss4.org/dr17/irspec/aspcap/
    '''

