import numpy as np 

def numeric_indices_generator(data, labels_dict):
    '''Generator function to get indices of numeric data in range.'''
    if data is not None:
        for label in labels_dict:
            interval = labels_dict[f'{label}']
            yield np.bitwise_and((interval[0] < data[f'{label}']),
                                 (data[f'{label}'] < interval[1]))

def field_indices_generator(data, fields_list):
    '''Generator function to get indices of data in correct field.'''
    if data is not None:
        for field in fields_list:
            yield data['field'] == field

def field_sifter(data, fields_list):
    '''
    Function that returns all records with correct field value.
    
    Returns all records in fits data with correct field values by using an
    iterator over the fields.
    
    Parameters
    ----------
    data : .fits file data
    fields_list : array_like
    
    Returns
    -------
    sifted_data : all .fits file records with a field in fields_list
    '''

    # Initializes iterator
    master_indices_list = []
    field_iterator = field_indices_generator(data, fields_list)
    init_bool = True
    for field in fields_list:
        # Checks to see if master_indices_list needs populating
        if init_bool:
            master_indices_list = next(field_iterator)
            init_bool = False
        # Gets indices for next field if master_indices_list is populated
        try:
            master_indices_list = np.bitwise_or(master_indices_list, next(field_iterator))
        except StopIteration:
            return data[master_indices_list]

def numeric_sifter(data, desired_values):
    '''
    Function that returns all records with correct numeric value.
    
    Returns all records in fits data with correct numeric values by using an
    iterator over the desired_values dictionary.
    
    Parameters
    ----------
    data : .fits file data
    desired_values : dictionary
    
    Returns
    -------
    sifted_data : all .fits file records with correct numeric values
    '''

    # Initializes iterator
    master_indices_list = []
    numeric_iterator = numeric_indices_generator(data, desired_values)
    init_bool = True
    for label in desired_values:
        # Checks to see if master_indices_list needs populating
        if init_bool:
            master_indices_list = next(numeric_iterator)
            init_bool = False
        # Gets indices for next label if master_indices_list is populated
        try:
            master_indices_list = np.bitwise_and(master_indices_list, next(numeric_iterator))
        except StopIteration:
            return data[master_indices_list]