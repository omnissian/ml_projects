import numpy as np
import os
import glob
import codecs
import re
import random
import math



#----------------------------------------
# x = np.array([[2], [4], [6], [8]])
# y = np.array([[0, 1, 2, 3]])
# print ("print (x)",x)
# print ("print (x.shape) ", x.shape) #x.shape = 4,1
# print ("print (y)", y)
# print ("print (y.shape)", y.shape) # y.shape = 1,4
# print (x.dot(y))
# print (y.dot(x))
#----------------------------------------

def parser (path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
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
                    example=np.append(example,1)
                else:
                    # example.append(0)
                    example=np.append(example,0)
        data_set.append(example)
        file_obj.close()
    return data_set

def actv_func_logist(neuron_in):
    neuron_in=neuron_in*(-1)
    neuron_out= ((1.0)/(1.0+math.exp(neuron_in)))
    return neuron_out

def cost_func_square(output, label ):

    return ((0.5)*(output-label)**2)

data_set=parser()
# input_data=np.array(data_set, float)
# print("shape (input_data) ", input_data.shape)
input_data=np.asarray(data_set,float)
#-------FORWARD PROPAGATION---------- labels for data have const values and manual input
M=1
N=3
O=1
# count of hidden layers = M
# count of neurons at one hidden layer = N
# count of neurons at output layer = O
set_size,inputs=input_data.shape # set_size-count (number) of all examples, inputs - var for count of inputs
labels=np.ones(set_size, float)  # labels for data
cumulative_error= 0.0

# in_W=np.full((inputs,1),float) # "good" array but how should i initialize it?
# in_W=np.ndarray.fill(random.randrange(0, 1)) # doesnt work
#--!!!WARNING!!!! if here is only one Hidden layer, then here will be only input matrix of weights and output!!!
#---!!!!! two matrices of weights total----------
in_W=np.random.rand(inputs, N)

# matrix of weights between inputs and first hidden layer
# shape: d= inputs, neurons = N
# q,b=input_data.shape # q= 10, b= 100, q-number of examples, b - number of "inputs"
out_W=np.random.rand(N,O)
# matrix of weights between last hidden layer and (number=1=O) output(s)
# initializing 3D matrix for weights
# array(whole network) of 2D matrices (W)
# every matrix represents connections from previous layer to current
inD,in_M=in_W.shape
outD,out_M=out_W.shape
# W=np.vstack((in_W,out_W)) we can't stack because all the input array dimensions...
#... except for the concatenation axis must match exactly
comp_matrix_tmp=np.array([],float)
label_of_exemplar = 0 #
for exemplar in input_data:
    # error_example=0.0
    # ind_d, ind_n=exemplar.shape # shape = (100,)
    comp_matrix_tmp=np.array((np.matmul(exemplar,in_W))[np.newaxis, :])
    #multiplied by link's weights between input and first layer in "hidden layer"
    cmpmat_D,cmpmat_N=comp_matrix_tmp.shape # (1, 3)
    print (comp_matrix_tmp.shape) #(1, 3)
    # maybe here we can use second loop (loop in loop) for multiply "input values" (row vector)
    # by a vector_n in matrix of weights gathered all together as a matrix of weights
    # dimension of a matrix D=number of neurons in each  layer, M= number of layers in "hidden layer"
    # M - could be here instead of zero
    for pos in range (cmpmat_N):
        comp_matrix_tmp[0][pos]=actv_func_logist(comp_matrix_tmp[0][pos])
    # in this loop we use activation func for every neuron input
    # matrix "comp_matrix_tmp" - collect values output values from previous layer

    # ---OUTPUT of neural network---------
    if(O>1):
        pass
        # here we must add some code *
    else:
        comp_matrix_tmp=np.matmul(comp_matrix_tmp,out_W)
        comp_matrix_tmp[0][0]=actv_func_logist(comp_matrix_tmp[0][0])
        error_example=cost_func_square(comp_matrix_tmp[0][0], labels[label_of_exemplar])
        # cumulative_error+=comp_matrix_tmp[0][0]
        cumulative_error+=error_example
    label_of_exemplar+=1

    # # --------------------------------------
    # for neuron_in in comp_matrix_tmp[0]:
    #     neuron_in=(actv_func_logist(neuron_in))
    # for neuron_in in enumerate(comp_matrix_tmp):
    #     neuron_in=(actv_func_logist(neuron_in))
    # for neuron_in in np.nditer(comp_matrix_tmp,op_flags=['readwrite']):
    #     neuron_in_val=float(neuron_in)
    #     neuron_in=(actv_func_logist(neuron_in_val))
    # # ---------------workable------------------------------
    # for neuron_in in np.nditer(comp_matrix_tmp,op_flags=['readwrite']):
    #     # neuron_in_val=float(neuron_in)
    #     neuron_in[...]=(actv_func_logist(neuron_in))
    # # ---------------workable------------------------------



    # check_matrix=exemplar.dot(in_W)
    # print(exemplar.shape)
    # print("print(type(comp_matrix_tmp))",type(comp_matrix_tmp))
    # print("print(comp_matrix_tmp.shape) ", comp_matrix_tmp.shape)

    pass
# W=np.array()
# W=np.append((in_W,out_W))

# W=np.append(in_W)

# W=[]
# W.append([])
# W.append(in_W)
# W.append(out_W)
# wD,wN,wN=W
# for m in M:
#     for N

#-----------------END---FORWARD PROPAGATION ---END-----------------

#-----------------------------------
print("end1")
print("end 2")









