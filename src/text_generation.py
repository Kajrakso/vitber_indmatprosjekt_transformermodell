from utils import onehot
import numpy as np
from data_generators import text_to_training_data
from train_test_params import *
from train_network import init_neural_network
import train_network
from layers_numba import CrossEntropy
import dill as pickle

def generate(net,start_idx,m,n_max,n_gen):
    
    #We will concatenate all generated integers (idx) in total_seq_idx
    total_seq_idx = start_idx

    n_total = total_seq_idx.shape[-1]
    slice = 0

    x_idx = start_idx

    while n_total < n_gen:
        n_idx = x_idx.shape[-1]
        X = onehot(x_idx,m)

        #probability distribution over m characters
        Z = net.forward(X)

        #selecting the last column of Z (distribution over final character)
        hat_Y = Z[0,:,-1]

        #sampling from the multinomial distribution
        #we do this instead of argmax to introduce some randomness
        #avoiding getting stuck in a loop
        y_idx = np.argwhere(np.random.multinomial(1, hat_Y.T)==1)

        if n_idx+1 > n_max:
            slice = 1

        #we add the new hat_y to the existing sequence
        #but we make sure that we only keep the last n_max elements
        x_idx = np.concatenate([x_idx[:,slice:],y_idx],axis=1)

        #we concatenate the new sequence to the total sequence
        total_seq_idx = np.concatenate([total_seq_idx,y_idx],axis=1)

        n_total = total_seq_idx.shape[-1]

    return total_seq_idx


def load_from_pkl_and_gen_text(filename:str) -> None:
    net = init_neural_network(text_params)
    net.load(filename)

    #We can now generate text from an initial string
    start_text = "Thou shall not"
    start_idx = np.array([text_to_idx[ch] for ch in start_text])[None]

    #length of the total text sequence we want to generate
    n_gen = 10*text_params.n_max

    generated_idx = generate(net,start_idx,m,text_params.n_max,n_gen)

    text = ''.join([idx_to_text[idx] for idx in generated_idx[0]])
    print(text)


def train():
    loss = CrossEntropy()
    L = train_network.train_network(
        network=net,
        x_train=np.array(data["x_train"]),
        y_train=np.array(data["y_train"]),
        loss_func=loss,
        alpha=text_params.alpha,
        n_iter=text_params.n_iter,
        num_ints=text_params.m,
        dump_to_pickle_file=True,
        file_name_dump="nn_dump_text_generation.pkl",
    )

    L.dump("nn_dump_text_generation_L.pkl")


if __name__ == "__main__":
    text_params = TextGenParams()

    with open('input.txt', 'r') as f:
        text = f.read()
        data,idx_to_text,text_to_idx, m = text_to_training_data(text_params.n_max,text,num_batches=text_params.b_train,batch_size=text_params.D)

        print("We will train on %d batches of size %d" % (len(data['x_train']),len(data['x_train'][0])))
        print("Each sequence has length %d" % text_params.n_max)

        print("Example of a sequence (chars): \n")
        print(''.join([idx_to_text[i] for i in data['x_train'][0][0]]))

        print("\nExample of a sequence (idx): \n")
        print(data['x_train'][0][0])

        text_params.m = m
    
    net = init_neural_network(text_params)


    train()
    # load_from_pkl_and_gen_text("nn_dump_text_generation.pkl")