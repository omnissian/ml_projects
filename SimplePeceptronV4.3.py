import numpy as np
import os
import glob
import codecs
import re
import random
import math


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
def parser(path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
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
def square_error(delta):
    return ((0.5)*((delta)**2))


def d_cost_sq(output, label):
    # output

    return 0


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
def create_labels_of_in_data(var=0, size=0, type=1, path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
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

in_data_set = np.array(parser(), dtype=float)
count_of_examples, exmpl_vector_d_size = in_data_set.shape
# exmpl_vector_d_size #row vector of input data
in_N, hidden_layers_count, neur_in_hid_l_count, neruons_out_l = 100, 10, 100, 1
in_W = np.random.rand(exmpl_vector_d_size, in_N)
hid_W = np.random.rand(hidden_layers_count, neur_in_hid_l_count, neur_in_hid_l_count)
out_W = np.random.rand(neur_in_hid_l_count, neruons_out_l)
# out_vals=np.array([])
out_vals = []
# --------------------------forward pass-----------------------------
for i in range(count_of_examples):
    forward_values = np.array([in_data_set[i]])  # row vector (1, 100)
    forward_values = np.matmul(forward_values, in_W)
    layer_activation(forward_values)
    counter = 0
    for layer in hid_W:
        forward_values = np.matmul(forward_values, layer)
        layer_activation(forward_values)
    forward_values = np.matmul(forward_values, out_W)
    layer_activation(forward_values)
    if (neruons_out_l == 1):
        out_vals.append(forward_values[0][0])
list_labels=create_labels_of_in_data(1, len(out_vals), float,) #vector, which holds labels for each example
out_vals=np.asarray([out_vals])
out_vals=out_vals.T # vector, which holds output of a network for each example
# --------------------------forward------------------------------------
delta_out=np.zeros((out_vals.shape))
for i in range(count_of_examples):
    for j in range (neruons_out_l):
        delta_out[i][j]=out_vals[i][j]-list_labels[i][j]

# --------------------evaluating cost function-------------------------
cost_delta_out=np.zeros((delta_out.shape))
for i in range(count_of_examples):
    for j in range (neruons_out_l):
        cost_delta_out[i][j]=square_error(delta_out[i][j])


print("empr")
print("gd")
