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

def forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set, labels):
    i = in_data_set.reshape(input_size, 1)
    i_transposed = i.T
    tmp=np.matmul(i_transposed, W_in)
    layer_activ(tmp,nHL) # choose the type of "activation function" in the layer_activ function
    H_in[:]=tmp.T #update the original
    tmp=np.matmul(H_in.T, W_hid[0])
    # H_hid[:, 0] = np.matmul(H_in, W_hid[0])
    tmp=tmp.T
    H_hid[:,0]=tmp[:,0]
    layer_activ(H_hid[:, 0], nHL)

    H_in = H_in.reshape(nHL, 1)
    return 1


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
# error = forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set[exmpl], labels[exmpl])
error = forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set[0], labels[0])
print("da")
print(H_in)

# # ------------END----neural network activation----END-------------
#
#
# # ------------------learning----------------------------------
# def layer_activation_sigma(H_vector):
#     # d,_n=H_vector.shape
#     # for neruon in range(d):
#     #     for layer_width in range(_n):
#     #         H_vector[neruon][layer_width]=((1.0)/(1.0+(math.exp((-1.0)*(H_vector[neruon][layer_width])))))
#     # d=int(H_vector.shape)
#     a = H_vector[np.newaxis, ...]  #
#     n, d = np.shape(a)  # n=1
#     for neruon in range(d):
#         tmp_activ = float(((1.0) / (1.0 + (math.exp((-1.0) * (H_vector[neruon]))))))
#         H_vector[neruon] = float(((1.0) / (1.0 + (math.exp((-1.0) * (H_vector[neruon]))))))
#     return
#     # pass
#
#
# # -----------full cycle learn--------------
# def forward_and_backprop(in_vector_row, in_W, in_N, hid_W, hidden_layers_count, neur_in_hid_l_count, out_W, neurons_out_l,
#                          desired_val):  # , H_hid,H_in, H_out - we dont need it because they are just temporary structures
#
#     # --------forward-------------
#     inNeur,connects=in_W.shape
#     in_vector_row=in_vector_row.reshape((inNeur,1))
#     in_vector_row=in_vector_row.T
#     # H_hid = np.array((hidden_layers_count, neur_in_hid_l_count),float)
#     H_hid = np.ones((hidden_layers_count, neur_in_hid_l_count), float)
#     in_H=np.matmul(in_vector_row,in_W)
#     H_hid[:][0] = np.matmul(in_H, in_W)
#     in_H=in_H.T
#     # # layer_activation_sigma(H_hid[:][0])
#     # chk_vector1=[H_hid[0][:]]
#     # chk_vector2=np.array([H_hid[0][:]])
#     # chk_vector3=H_hid[0]
#     # print("hello")
#     # # layer_activation_sigma(H_hid[:][0])
#     layer_activation_sigma(H_hid[0])
#
#     for nlayer in range(1, hidden_layers_count):
#         H_hid[:][nlayer] = np.matmul((H_hid[:][nlayer - 1]).T, hid_W[nlayer])
#         # H_hid[nlayer] = np.matmul((H_hid[:][nlayer - 1]).T, hid_W[nlayer])
#         layer_activation_sigma(H_hid[nlayer])
#
#     H_out = np.matmul([H_hid[hidden_layers_count - 1]], out_W)
#     layer_activation_sigma(H_out[0])
#
#     # if(neruons_out_l==1):
#     #     # H_out = float(np.matmul(H_hid[hidden_layers_count - 1], out_W))
#     #     H_out = np.matmul(H_hid[hidden_layers_count - 1], out_W)
#     #     # tmp=H_out[0]
#     #     H_out=layer_activation_sigma(H_out)
#     # else:
#     #     pass
#
#     # print("start")
#     # H_out = np.matmul((np.array([[nlayer - 1]]).T), out_W)
#     # leftMatrx1=np.array([H_hid[hidden_layers_count - 1]]) #shape = 1, 100
#     # leftMatrx2=np.array([H_hid[hidden_layers_count - 1]]).T
#     # H_out = np.matmul((np.array([[hidden_layers_count - 1]]).T), out_W)
#     # H_out = np.matmul((np.array([H_hid[hidden_layers_count - 1]])), out_W)
#     # H_out = np.matmul(H_hid[hidden_layers_count - 1], out_W)
#     # H_out[0]=layer_activation_sigma(H_out[0]) # NaN, WTF??? but when we do stuff the function debuger shows that it has been changed (global value). WTF
#     # H_out[0][0] = layer_activation_sigma(H_out[0]) #NaN too
#     # tmp_val=float(((1.0) / (1.0 + (math.exp((-1.0) * (H_out[0][0]))))))
#     # H_out[0][0]=tmp_val
#     # H_out[0][0]== float(((1.0) / (1.0 + (math.exp((-1.0) * (H_out[0][0]))))))
#
#     # layer_activation_sigma(H_out[0])
#     print("end forward")
#     # --------forward END-------------
#     error = np.array([[]])  # this error we will use for backpropagation through each layer
#     # pass
#
#     # --------forward END-------------
#     # ----------backpropagation----------------
#
#     if (neruons_out_l == 1):
#         pass
#         error =  H_out[0][0]-desired_val
#         # error_global=np.copy(error)
#     else:
#         pass
#     # error= # finish this stuff, for ouptut layer if neurons count bigger than 1, error will be array(vector)
#
#     # square error
#     sq_error_val = square_error(error)
#     # update weights (matrix of weights) between neuron(s) in last(output) layer and last layer in hidden layer
#     update_grad_g=[]
#     # update_weights(out_W, H_out,error, update_grad_g)
#
#
#     update_weights( in_vector_row,in_W, in_H, hid_W, H_hid, out_W, H_out,error,neruons_out_l)
#
#     return
#
#
# # ----------backpropagation----------------
# x = np.ones((3, 3, 3), float)
#
#
# # ---------backprop func----------
# # update_grad=np.copy(W)
# # def update_weights(in_W, in_H, in_update_grad, hid_W, hid_H, hid_update_grad, out_W, out_H, out_update_grad,E):  # W- matrix of weights that we want to update 2D matrix!!! ,
# def update_weights(in_vector_row, in_W, in_H, hid_W, hid_H, out_W, out_H, E, neruons_out_l):
#     layers_in_hid, neur_in_last_hid_l=hid_H.shape
#     ns_in_out,d=out_H.shape #ns_in_out-neurons in out layer, d- dimension
#     error = np.reshape(E, (ns_in_out, 1))
#     # error_new=np.matmul(hid_W[layers_in_hid-1],error)
#     upd_out_W=np.copy(out_W)
#     error_new=np.matmul(out_W,error) #error propagated for each neuron in previous layer !ERROR for PREVIOUS LAYER DONE!
#     for n_current_l in range(ns_in_out):
#         for n_previous_l in range(neur_in_last_hid_l):
#             # var1=error[n_current_l][0]
#             # var2=out_H[n_current_l][0]*(1.0-out_H[n_current_l][0])
#             # var3=hid_H[layers_in_hid-1][n_previous_l]
#             upd_out_W[n_previous_l][n_current_l]=error[n_current_l][0]*out_H[n_current_l][0]*(1.0-out_H[n_current_l][0])*hid_H[layers_in_hid-1][n_previous_l]
#
#     out_W=out_W-upd_out_W    # update matrix of weights for connection between hid layer and output layer
#
#     upd_hid_W=np.copy(hid_W)
#     last_layer_hid=layers_in_hid-1
#     # checker =0
#     for layer in range(last_layer_hid,0,-1):
#         error = np.copy(error_new)
#         error_new = np.matmul(hid_W[layer],error)
#         for n_current_l in range(neur_in_last_hid_l):
#             for n_previous_l in range(neur_in_last_hid_l):
#                 value=error[n_current_l][0]*hid_H[layer][n_current_l]*(1.0-hid_H[layer][n_current_l])*hid_H[layer-1][n_previous_l]
#                 # upd_hid_W[layer][n_previous_l][n_current_l]=error[n_current_l][0]*hid_H[layer][n_current_l]*(1.0-hid_H[layer][n_current_l])*hid_H[layer-1][n_previous_l]
#                 upd_hid_W[layer][n_previous_l][n_current_l]=value
#                 # checker=layer-1
#     #----first layer in hidden layer connected with outputs from input layer (input layer have neurons)
#     error = np.copy(error_new)
#     error_new = np.matmul(hid_W[0], error)
#     n_previous_l_count,d=in_H.shape
#
#     for n_current_l in range(neur_in_last_hid_l):
#         for n_previous_l in range(n_previous_l_count):
#             # value = error[n_current_l][0] * hid_H[0][n_current_l] * (1.0 - hid_H[0][n_current_l]) * H_in[] \
#             #         hid_H[0][n_previous_l]
#             # upd_hid_W[layer][n_previous_l][n_current_l]=error[n_current_l][0]*hid_H[layer][n_current_l]*(1.0-hid_H[layer][n_current_l])*hid_H[layer-1][n_previous_l]
#             value = error[n_current_l][0] * hid_H[0][n_current_l] *(1.0-hid_H[0][n_current_l])*in_H[n_previous_l][0]
#             upd_hid_W[0][n_previous_l][n_current_l] = value
#
#     hid_W=hid_W-upd_hid_W    #update matrix of weights in hidden layer
#
#     error = np.copy(error_new)
#     error_new = np.matmul(in_W, error)
#     neur_in_last_l,d=in_H.shape
#     n_previous_l_count,d=in_vector_row.T.shape
#     upd_in_W=np.copy(in_W)
#     for n_current_l in range(neur_in_last_l):
#         for n_previous_l in range(n_previous_l_count):
#             # value = error[n_current_l][0] * hid_H[0][n_current_l] * (1.0 - hid_H[0][n_current_l]) * H_in[] \
#             #         hid_H[0][n_previous_l]
#             # upd_hid_W[layer][n_previous_l][n_current_l]=error[n_current_l][0]*hid_H[layer][n_current_l]*(1.0-hid_H[layer][n_current_l])*hid_H[layer-1][n_previous_l]
#             value = error[n_current_l][0] * in_H[n_current_l] *(1.0-in_H[n_current_l])*in_vector_row[0][n_previous_l]
#             upd_in_W[n_previous_l][n_current_l] = value
#     in_W=in_W-upd_in_W # update WEIGHTS but you forgot the LEARNING RATE!!!
#
#     print("da")
#     print("cheker= ")
#     print("go")
#
#
#
#     # # H-matrix of output values from neurons in current layer,
#     # # E-matrix of error for each connection between neuron in previous layer and neuron in current layer
#     # dW, nnW = out_W.shape
#     # dl, nn = out_H.shape  # dl-count of layers, nn-neurons in current layer
#     # #dl for out layer is a number of neurons in output layer, nn- dimensions of output layer
#     # # dE,nE=E.shape # dE should equals to nn - neurons in current layer (error for each layer), nE==1 just a dimension
#     # out_H=np.reshape(nn,dl)
#     # # error = [[]]
#     # error=E
#     # error=np.reshape(E,(nnW,1))
#     # error_new = [[]]
#     # update_grad_out=np.copy(out_W)
#     # dliter=dl-1
#     # #---------update weights between last layer in hidden layer and output layer---------
#     # if(neruons_out_l==1):
#     #     layers_in_hid_layer, neurons_in_hid_l_l=hid_H.shape
#     #     error_new=np.matmul(out_W,error)
#     #     for neuron_current in range(nn): #nn=1, dl=1 - for output layer
#     #         error_new=np.matmul(out_W,error)
#     #         test1=out_H[neuron_current][0]
#     #         typer=out_H
#     #         # update_grad_out[neuron_previous][neuron_current]=error[neuron_previous][neuron_current]*out_H[neuron_current][0]*(1.0-out_H[neuron_current][0])*hid_H[layers_in_hid_layer-1][neuron_previous]
#     #         out_W=out_W-update_grad_out
#     #         #---update weights-----
#     #         layers_in_hid, neurons_in_hid_layer=hid_H.shape
#     #
#     #         # for neruon in range(neurons_in_hid_layer):
#     #         error_new=np.matmul(out_W,error)
#     #         update_grad_out
#     #         update_grad_out[previous][current] = np.matmul(hid_W[size_of_out_W - 1], error)
#     #         # update_grad_out[previous][current] = np.matmul(hid_W[size_of_out_W - 1], error)
#     #
#     #
#     #         #---update weights-----
#     #
#     # else:
#     #     pass
#     #     #here need to write the same code as above, but the error will be represented as a vector of values for each error in output neuron in output layer
#     #
#     # # ---------update weights between last layer in hidden layer and output layer---------
#
# #------------------------just a paste backup-----------------------------------------
#     # if(neruons_out_l==1):
#     #     layers_in_hid_layer, neurons_in_hid_l_l=hid_H.shape
#     #     error_new=np.matmul(out_W,error)
#     #     for neuron_current in range(nn):
#     #         for neuron_previous in range(dW):
#     #             test1=out_H[neuron_current][0]
#     #             typer=out_H
#     #             # update_grad_out[neuron_previous][neuron_current]=error[neuron_previous][neuron_current]*out_H[neuron_current][0]*(1.0-out_H[neuron_current][0])*hid_H[layers_in_hid_layer-1][neuron_previous]
#     #
#     #     out_W=out_W-update_grad_out
#     # else:
#     #     pass
#     #     #here need to write the same code as above, but the error will be represented as a vector of values for each error in output neuron in output layer
# #----------------------just a paste backup------------------------
#
#     #
#     # for layer in range(dliter):
#     #     for neuron_current in range(nn):
#     #         for neuron_previous in range(dW):
#     #             # update_grad[layer][neuron_previous][neuron_current]=error[neuron_previous][neuron_current]*H[dl-layer][neuron_current]*(1.0-H[dl-layer][neuron_current])*
#     #             pass
#     # pass
#     return
#
#
# # ---------backprop func----------
#
# # -----------full cycle learn--------------
# # -----checker---------
#
# label = np.array([[1]], float)
# label[0][0] = 1
# forward_and_backprop(in_data_set[0][:], in_W, in_N, hid_W, hidden_layers_count, neur_in_hid_l_count, out_W,
#                      neruons_out_l,
#                      label[0][0])
#
# update_weights(out_W,H_out,)
#
# # -------------------back propagation------------------------------------------
#
#
# # --------------------------forward pass-----------------------------
# # -----------------------------------------------------------------------------------------------------------
# #
# # for i in range(count_of_examples):
# #     forward_values = np.array([in_data_set[i]])  # row vector (1, 100)
# #     forward_values = np.matmul(forward_values, in_W)
# #     layer_activation(forward_values)
# #     counter = 0
# #     for layer in hid_W:
# #         forward_values = np.matmul(forward_values, layer)
# #         layer_activation(forward_values)
# #     forward_values = np.matmul(forward_values, out_W)
# #     layer_activation(forward_values)
# #     if (neruons_out_l == 1):
# #         out_vals.append(forward_values[0][0])
# # list_labels = create_labels_of_in_data(1, len(out_vals), float, )  # vector, which holds labels for each example
# # out_vals = np.asarray([out_vals])
# # out_vals = out_vals.T  # vector, which holds output of a network for each example
# # # ------------------------end--forward pass---------------------------------
# # delta_out = np.zeros((out_vals.shape))
# # for i in range(count_of_examples):
# #     for j in range(neruons_out_l):
# #         delta_out[i][j] = out_vals[i][j] - list_labels[i][j]
# #
# # # --------------------evaluating cost function-------------------------
# # cost_delta_out = np.zeros((delta_out.shape))  # vector of (1/2)*(output-desired output)**2 for every example
# # for i in range(count_of_examples):
# #     for j in range(neruons_out_l):
# #         cost_delta_out[i][j] = square_error(delta_out[i][j])
# #
# # avg_cost = (cost_delta_out.sum() / count_of_examples)  # scalar of mean error through given data set
# #
# # # ------------------end---evaluating cost function-------------------------
# #
# #
# # upd_out_W = np.zeros(out_W.shape)
# # upd_hid_W = np.zeros(hid_W.shape)
# # upd_in_W = np.zeros(in_W.shape)


# ------------------------------------------------------------------------------------------------------------


print("empr")
print("gd")
