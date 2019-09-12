import numpy as np
import os
import glob
import codecs
import re
import random
import math


# change parser a little bit, so he can read the input file and label from it, and after convey this label through all operations

# add a learning rate
# add cycle mechanism for learn a network in loop (add output of current error of  each iteration)
# maybe rewrite whole code
# if rewrite - draw a block-scheme of each matrix of weight, and H(output value matrix) for each layer for clear and PURE understanding


# ----------------------------------------
# x = np.array([[2], [4], [6], [8]])
# y = np.array([[0, 1, 2, 3]])
# print ("print (x)",x)
# print ("print (x.shape) ", x.shape) #x.shape = 4,1
# print ("print (y)", y)
# print ("print (y.shape)", y.shape) # y.shape = 1,4
# print (x.dot(y))
# print (y.dot(x))
# ----------------------------------------


def parser(labels, path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
    data_set = []
    for filename in os.listdir(path):
        tmp_path = path + filename
        file_obj = codecs.open(tmp_path, 'r')
        list_obj = file_obj.readlines()
        example = np.array([])
        tmp_splitted = []
        # print(list_obj)
        for string_n in list_obj:
            tmp = re.split('[^0-1]', string_n)
            tmp_splitted.append(tmp[0])
        for line_n in tmp_splitted:
            for char_n in line_n:
                if ((int(char_n))):
                    # example.append(1)
                    example = np.append(example, 1)
                else:
                    # example.append(0)
                    example = np.append(example, 0)

        lgth_e = (len(example)) - 1
        labels.append(example[lgth_e])
        example = np.delete(example, lgth_e, 0)
        data_set.append(example)
        file_obj.close()
    return data_set


def actv_func_logist(neuron_in):
    neuron_in = neuron_in * (-1)
    neuron_out = ((1.0) / (1.0 + math.exp(neuron_in)))
    return neuron_out


def cost_func_square(output, label):
    return ((0.5) * (output - label) ** 2)
    # in the func below - d_cost_sq
    # output is the sum of all outputs
    # label is the sum of all labels
    # ?question is how to represent as a list\ and where to calculate the mean values?


def square_error(delta): #both arguments are scalars
    return ((0.5) * ((delta) ** 2))


def deriv_cost_mse(output, label): # all arguments are scalars
    # output
    return ((1.0*output)-label)


def changer(value):
    value[0][0] = 2  # изменит и оригинал (если это list) или numpy array, но не изменит int val
    # как понимаю передаются указатели массивов, но копии "базовых типов"


#
# c=np.ones((1,1),float)
# print (c)
# changer (c)
# print (c)
# b=23
# use first arg "1" for manual in (without read labels from a file), 0 for choose the
def create_labels_of_in_data(var=0, size=0, type=1,
                             path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
    if (var):
        pass
        if (size):
            list_of_labels = np.ones((size, 1), dtype=type)
            return list_of_labels
        else:
            pass
    else:
        pass


def layer_activation(forward_values):
    b, fr_vals_size = forward_values.shape
    for i in range(fr_vals_size):
        forward_values[0][i] = actv_func_logist(forward_values[0][i])


def activ_func(scalar_value):
    scalar_value = scalar_value * (-1)
    scalar_value = ((1.0) / (1.0 + math.exp(scalar_value)))
    return scalar_value


def layer_activ(row_vector, count_of_neurons):
    for n in range(count_of_neurons):
        row_vector[0, n] = activ_func(row_vector[0, n])


# ----------------neural network activation---------------
# exmpl_vector_d_size #row vector of input data
# in_N, hidden_layers_count, neur_in_hid_l_count, neruons_out_l = 100, 10, 100, 1
# in_W = np.random.rand(exmpl_vector_d_size, in_N)
# hid_W = np.random.rand(hidden_layers_count, neur_in_hid_l_count, neur_in_hid_l_count)
# out_W = np.random.rand(neur_in_hid_l_count, neruons_out_l)
# # out_vals=np.array([])
# # out_vals = []
# H_hid = np.array([])
# H_in = np.array([])
# H_out = np.array([])

# last argument in forward tells us should we return error or output value
# type=0 for output value (for check)
#default type =1 for error (for future learn)
def forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set, label,out_size,type=1):
    i = in_data_set.reshape(input_size, 1) #just a data of one example
    i_transposed = i.T
    tmp=np.matmul(i_transposed, W_in)
    layer_activ(tmp,nHL) # choose the type of "activation function" in the layer_activ function
    H_in[:]=tmp.T #update the original
    tmp=np.matmul(H_in.T, W_hid[0])
    # H_hid[:, 0] = np.matmul(H_in, W_hid[0])
    layer_activ(tmp,nHL)
    tmp=tmp.T
    H_hid[:,0]=tmp[:,0]
    for i in range(1,HL):
        tmp=tmp.T # for make it row vector again
        Wtmp=W_hid[i]
        tmp=np.matmul(tmp,W_hid[i])
        layer_activ(tmp,nHL)
        tmp=tmp.T # for make it column vector
        H_hid[:,i]=tmp[:,0]
    tmp=tmp.T # for make it row vector again
    tmp=np.matmul(tmp,W_out)
    layer_activ(tmp,out_size)
    H_out[:]=tmp
    # error=cost_func_square()
    if(type):
        if(out_size==1):
            print (cost_func_square(H_out[0][0],label))
            return deriv_cost_mse(H_out[0][0],label) #choose the cost function and her derivative
        else:
            pass # for output layer which one have more than one neuron
            return 0
    else:
        if (out_size == 1):
            print (H_out[0][0])
            return H_out[0][0]  # choose the cost function and her derivative

        else:
            pass  # for output layer which one have more than one neuron
            return 0

def back_prop(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set, label,out_size):
    



labels = []
# parsers_out=parser(labels)
in_data_set = np.array(parser(labels),
                       dtype=float)  # for 1 d output, or scalar output !!! so the neurons in output layer=1!!!
count_of_examples, exmpl_vector_d_size = in_data_set.shape
input_size = exmpl_vector_d_size  # number of inputs "of data", dimension of vector i
out_size = 1  # equal the size of neurons in ouput layer, also depends on parser that you chose

# ----input layer (connection between input layer and first layer in hidden layer)----------
HL = 10  # number of hidden layers
nHL = 100  # number of neurons in one hidden layer
W_in = np.random.rand(input_size, nHL)  # weights
H_in = np.zeros((nHL, 1))  # vector E R^(nHLx1)
# ----------------hidden layer----------------
W_hid = np.random.rand(HL, nHL, nHL)  # weights
H_hid = np.zeros((nHL, HL))  # matrix ofr output values for each neuron in hidden layers layer
# ----------output layer----------
W_out = np.random.rand(nHL, out_size)  # weights
H_out = np.zeros((out_size, 1))
# -------------------------------------------
learning_rate = 0.002  # just a constant value for a while, maybe add a "momentum" in the fьюча or would make a custom "changer" for it

iterations = 10 ** 7  # number of iterations
#
# for iteration in range(iterations):
#     for exmpl in range(count_of_examples):
# error = forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set[exmpl], labels[exmpl],out_size)
error = forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set[0], labels[0],out_size)
print("da")
# print(H_in)


print("empr")
print("gd")
