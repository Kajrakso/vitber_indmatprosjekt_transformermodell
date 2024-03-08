def get_training_params_sort() -> dict:
    """training parameters as per part two of excersise 3.3 in proj. desc."""
    training_params = {
        'D': 250,       # number of datapoint (x, y)
        'b_train': 10,  # number of batches to train on
        'b_test': 10,   # number of batches to train on
        
        'r': 7,         # length of the sequences
        'n_max': 13,    # 2*r - 1
        'm': 5,         # number of symbols

        'd': 20,        # output dimension for the linear layer.
        'k': 10,        # dimension for attention step
        'p': 25,        # dimensions for feed forward
        'L': 2,         # number of transformer layers

        'alpha': 0.001, # alpha paramater for Adam
        'n_iter': 300   # number of iterations
    }
    return training_params


def get_training_params_addition() -> dict:
    """training parameters as per excersise 3.4 in proj. desc."""
    params = {
        'D': 250,       # number of datapoint (x, y)
        'b_train': 20,  # number of batches to train on
        'b_test': 10,   # number of batches to test on
        
        'r': 2,         # number of digits
        'n_max': 6,     # 3 * r
        'm': 10,        # number of symbols (0, 1, ..., 9)

        'd': 30,        # output dimension for the linear layer.
        'k': 20,        # dimension for attention step
        'p': 40,        # dimensions for feed forward
        'L': 3,         # number of transformer layers

        'alpha': 0.001, # alpha paramater for Adam
        'n_iter': 150   # number of iterations
    }
    return params