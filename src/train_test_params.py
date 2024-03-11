from dataclasses import dataclass

@dataclass
class BaseParams():
    D = 250       # number of datapoint (x, y)
    b_train = 10  # number of batches to train on
    b_test = 10   # number of batches to train on
        
    r = None      # length of input sequence
    n_max = None  # max length of input sequence
    n_y = None    # length of output sequence
    
    m = None      # number of symbols

    d = None      # output dimension for the linear layer.
    k = None      # dimension for attention step
    p = None      # dimensions for feed forward
    L = None      # number of transformer layers

    alpha = 0.01  # alpha paramater for Adam
    n_iter = 300  # number of iterations

@dataclass
class SortParams1(BaseParams):
    r = 5
    n_max = 9       # 2*r - 1
    m = 2
    d = 10
    k = 5
    p = 15
    L = 2 

@dataclass
class SortParams2(BaseParams):
    r = 7
    n_max = 13      # 2*r - 1
    m = 5
    d = 20
    k = 10
    p = 25
    L = 2

@dataclass
class AddParams(BaseParams):
    b_train = 20
    r = 2
    n_max = 6       # 3*r
    m = 10
    d = 30
    k = 20
    p = 40
    L = 3
    n_iter = 150

@dataclass
class TextGenParams(BaseParams):
    D = 50
    b_train = 20
    b_test = 10
    r = 2
    n_max = 50
    m = 0
    d = 80
    k = 25
    p = 100
    L = 2
    n_iter = 150