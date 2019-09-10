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
    return ((0.5) * ((delta) ** 2))


def dif_cost_sq(output, label):
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


# ----------------neural network activation---------------
in_data_set = np.array(parser(), dtype=float)
count_of_examples, exmpl_vector_d_size = in_data_set.shape
# exmpl_vector_d_size #row vector of input data
in_N, hidden_layers_count, neur_in_hid_l_count, neruons_out_l = 100, 10, 100, 1
in_W = np.random.rand(exmpl_vector_d_size, in_N)
hid_W = np.random.rand(hidden_layers_count, neur_in_hid_l_count, neur_in_hid_l_count)
out_W = np.random.rand(neur_in_hid_l_count, neruons_out_l)
# out_vals=np.array([])
# out_vals = []
H_hid = np.array([])
H_in = np.array([])
H_out = np.array([])


# ------------END----neural network activation----END-------------


# ------------------learning----------------------------------
def layer_activation_sigma(H_vector):
    # d,_n=H_vector.shape
    # for neruon in range(d):
    #     for layer_width in range(_n):
    #         H_vector[neruon][layer_width]=((1.0)/(1.0+(math.exp((-1.0)*(H_vector[neruon][layer_width])))))
    # d=int(H_vector.shape)
    a = H_vector[np.newaxis, ...]  #
    n, d = np.shape(a)  # n=1
    for neruon in range(d):
        tmp_activ = float(((1.0) / (1.0 + (math.exp((-1.0) * (H_vector[neruon]))))))
        H_vector[neruon] = float(((1.0) / (1.0 + (math.exp((-1.0) * (H_vector[neruon]))))))
    return
    # pass


# -----------full cycle learn--------------
def forward_and_backprop(in_vector, in_W, in_N, hid_W, hidden_layers_count, neur_in_hid_l_count, out_W, neurons_out_l,
                         desired_val):  # , H_hid,H_in, H_out - we dont need it because they are just temporary structures

    # --------forward-------------
    # H_hid = np.array((hidden_layers_count, neur_in_hid_l_count),float)
    H_hid = np.ones((hidden_layers_count, neur_in_hid_l_count), float)
    H_hid[:][0] = np.matmul(in_vector, in_W)
    # # layer_activation_sigma(H_hid[:][0])
    # chk_vector1=[H_hid[0][:]]
    # chk_vector2=np.array([H_hid[0][:]])
    # chk_vector3=H_hid[0]
    # print("hello")
    # # layer_activation_sigma(H_hid[:][0])
    layer_activation_sigma(H_hid[0])

    for nlayer in range(1, hidden_layers_count):
        H_hid[:][nlayer] = np.matmul((H_hid[:][nlayer - 1]).T, hid_W[nlayer])
        # H_hid[nlayer] = np.matmul((H_hid[:][nlayer - 1]).T, hid_W[nlayer])
        layer_activation_sigma(H_hid[nlayer])

    H_out = np.matmul([H_hid[hidden_layers_count - 1]], out_W)
    layer_activation_sigma(H_out[0])

    # if(neruons_out_l==1):
    #     # H_out = float(np.matmul(H_hid[hidden_layers_count - 1], out_W))
    #     H_out = np.matmul(H_hid[hidden_layers_count - 1], out_W)
    #     # tmp=H_out[0]
    #     H_out=layer_activation_sigma(H_out)
    # else:
    #     pass

    # print("start")
    # H_out = np.matmul((np.array([[nlayer - 1]]).T), out_W)
    # leftMatrx1=np.array([H_hid[hidden_layers_count - 1]]) #shape = 1, 100
    # leftMatrx2=np.array([H_hid[hidden_layers_count - 1]]).T
    # H_out = np.matmul((np.array([[hidden_layers_count - 1]]).T), out_W)
    # H_out = np.matmul((np.array([H_hid[hidden_layers_count - 1]])), out_W)
    # H_out = np.matmul(H_hid[hidden_layers_count - 1], out_W)
    # H_out[0]=layer_activation_sigma(H_out[0]) # NaN, WTF??? but when we do stuff the function debuger shows that it has been changed (global value). WTF
    # H_out[0][0] = layer_activation_sigma(H_out[0]) #NaN too
    # tmp_val=float(((1.0) / (1.0 + (math.exp((-1.0) * (H_out[0][0]))))))
    # H_out[0][0]=tmp_val
    # H_out[0][0]== float(((1.0) / (1.0 + (math.exp((-1.0) * (H_out[0][0]))))))

    # layer_activation_sigma(H_out[0])
    print("end forward")
    # --------forward END-------------
    error = np.array([[]])  # this error we will use for backpropagation through each layer
    # pass

    # --------forward END-------------
    # ----------backpropagation----------------

    if (neruons_out_l==1):
        pass
        error=desired_val-H_out[0][0]
    else:
        pass
    # error= # finish this stuff, for ouptut layer if neurons count bigger than 1, error will be array(vector)

    #square error
    sq_error_val=square_error(error)
    #update weights (matrix of weights) between neuron(s) in last(output) layer and last layer in hidden layer




    return
# ----------backpropagation----------------
x=np.ones((3,3,3), float)


#---------backprop func----------

def update_weights(W,H,E): #W- matrix of weights that we want to update 2D matrix!!! ,
    # H-matrix of output values from neurons in current layer,
    #E-matrix of error for each connection between neuron in previous layer and neuron in current layer
    dW,nnW=W.shape
    dl,nn=H.shape #dl-count of layers, nn-neurons in current layer
    #dE,nE=E.shape # dE should equals to nn - neurons in current layer (error for each layer), nE==1
    error=[[]]
    for layer in range(dl):
        for neuron_current in range(nn):
            for neuron_previous in range(dW):



    pass
    return

#---------backprop func----------

# -----------full cycle learn--------------
# -----checker---------

label = np.array([[1]], float)
label[0][0] = 1
forward_and_backprop(in_data_set[0][:], in_W, in_N, hid_W, hidden_layers_count, neur_in_hid_l_count, out_W,
                     neruons_out_l,
                     label[0][0])

# -------------------back propagation------------------------------------------


# --------------------------forward pass-----------------------------
# -----------------------------------------------------------------------------------------------------------
#
# for i in range(count_of_examples):
#     forward_values = np.array([in_data_set[i]])  # row vector (1, 100)
#     forward_values = np.matmul(forward_values, in_W)
#     layer_activation(forward_values)
#     counter = 0
#     for layer in hid_W:
#         forward_values = np.matmul(forward_values, layer)
#         layer_activation(forward_values)
#     forward_values = np.matmul(forward_values, out_W)
#     layer_activation(forward_values)
#     if (neruons_out_l == 1):
#         out_vals.append(forward_values[0][0])
# list_labels = create_labels_of_in_data(1, len(out_vals), float, )  # vector, which holds labels for each example
# out_vals = np.asarray([out_vals])
# out_vals = out_vals.T  # vector, which holds output of a network for each example
# # ------------------------end--forward pass---------------------------------
# delta_out = np.zeros((out_vals.shape))
# for i in range(count_of_examples):
#     for j in range(neruons_out_l):
#         delta_out[i][j] = out_vals[i][j] - list_labels[i][j]
#
# # --------------------evaluating cost function-------------------------
# cost_delta_out = np.zeros((delta_out.shape))  # vector of (1/2)*(output-desired output)**2 for every example
# for i in range(count_of_examples):
#     for j in range(neruons_out_l):
#         cost_delta_out[i][j] = square_error(delta_out[i][j])
#
# avg_cost = (cost_delta_out.sum() / count_of_examples)  # scalar of mean error through given data set
#
# # ------------------end---evaluating cost function-------------------------
#
#
# upd_out_W = np.zeros(out_W.shape)
# upd_hid_W = np.zeros(hid_W.shape)
# upd_in_W = np.zeros(in_W.shape)


# ------------------------------------------------------------------------------------------------------------


print("empr")
print("gd")
