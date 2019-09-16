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

# D:\Polygon\Examples\plgn
# def parser(labels, path='C:\\Users\\Anatoly\\Desktop\\Work1\\plgn\\data10by10digitstest\\forlearn7\\'):
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


def cost_func_square(output, label):
    return ((0.5) * (output - label) ** 2)
    # in the func below - d_cost_sq
    # output is the sum of all outputs
    # label is the sum of all labels
    # ?question is how to represent as a list\ and where to calculate the mean values?


def square_error(delta):  # both arguments are scalars
    return ((0.5) * ((delta) ** 2))


def deriv_cost_mse(output, label):  # all arguments are scalars
    # output
    return ((1.0 * output) - label)


def changer(value):
    value[0][0] = 2  # изменит и оригинал (если это list) или numpy array, но не изменит int val
    # как понимаю передаются указатели массивов, но копии "базовых типов"



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

# def sigmoid(x):
# def activ_func(x):
#     # tmpX=x
#     if x < 0:
#         # tmp1=(1.0 - 1.0 / (1.0 + math.exp(x)))
#         # print("tmp1= ",tmp1)
#         return 1.0 - 1.0 / (1.0 + math.exp(x))
#     else:
#         # tmp2=(1.0 - 1.0 / (1.0 + math.exp(x)))
#         # print("tmp2= ",tmp2)
#         return 1.0 / (1.0 + math.exp(-x))

#
# def activ_func(scalar_value):
#     # # # print("scalar_value= ",scalar_value)
#     # # scalar_value = scalar_value * (-1)
#     # # scalar_value = ((1.0) / (1.0 + math.exp(scalar_value)))
#     # # # scalar_value = ((1.0) / (1.0 + 2.7182**(scalar_value)))
#     # # #---------below this point----https://stackoverflow.com/questions/36268077/overflow-math-range-error-for-log-or-exp?rq=1
#     # tmp = scalar_value
#     # if(scalar_value<0):
#     #     return ((1.0) / (1.0 + math.exp(tmp)))
#     # else:
#     #     return  ((1.0) / (1.0 + math.exp(tmp)))
#     # # return tmp
#     # # #----------------------------
#     # if (scalar_value < 0):
#     #     return (1.0 - 1.0 / (1.0 + math.exp(scalar_value)))
#     # return ((1.0) / (1.0 + math.exp(-scalar_value)))
#     # # #----------------------------
#     # if (scalar_value < 0):
#     #     return (1.0 - 1.0 / (1.0 + np.exp(scalar_value)))
#     # return ((1.0) / (1.0 + np.exp(-scalar_value)))
#     #----https://stackoverflow.com/questions/42524679/python-overflowerror-math-range-error------------
#     # tmp= scalar_value * (-1)
#     # if(tmp<(-9)):
#     #     tmp=-9
#     # else:
#     #     tmp=9
#     # # np.exp(np.array([1391.12694245], dtype=np.float128))
#     # # np.exp(np.array([tmp], dtype=np.float128))
#     # # tmp=np.exp(np.array([tmp], dtype=np.double))
#     # tmp=np.round((np.exp(tmp)) ,5)
#     # scalar_value = np.round(((1.0) / (1.0 + tmp)),5)
#     #--------------------------------------------
#     tmp=-1.0*scalar_value
#     tmp=math.exp(tmp)
#     tmp=1.0-tmp
#     tmp=1.0/tmp
#     print("before round tmp= ", tmp)
#     tmp=np.round(tmp,5)
#     print("after ROUND tmp= ", tmp)
#     #----checker end------------
#     scalar_value=-1.0*scalar_value
#
#
#     scalar_value=(1.0/(1.0-math.exp(scalar_value)))
#     return scalar_value

from math import exp

def activ_func(scalar_value):
    # # # np.exp(np.array([tmp], dtype=np.float128))
    # # tmp=(np.longlong(np.exp(np.array(-scalar_value,dtype=np.longlong))))
    # # teml=1.0/(1.0+tmp)
    # # tmp=2,7182818284**(-scalar_value)
    # tmp=2.71828**(-scalar_value)
    # teml=1.0/(1.0+tmp)
    # # return ((1.0)/((1.0)+tmp))

    return (1.0)/(1.0+exp(-scalar_value))


def layer_activ(row_vector, count_of_neurons):
    for n in range(count_of_neurons):
        row_vector[0, n] = activ_func(row_vector[0, n])


def forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set, label, out_size, type=1):
    i = in_data_set.reshape(input_size, 1)  # just a data of one example
    i_transposed = i.T
    tmp = np.matmul(i_transposed, W_in)
    layer_activ(tmp, nHL)  # choose the type of "activation function" in the layer_activ function
    H_in[:] = tmp.T  # update the original
    tmp = np.matmul(H_in.T, W_hid[0])
    layer_activ(tmp, nHL)
    tmp = tmp.T
    H_hid[:, 0] = tmp[:, 0]
    for i in range(1, HL):
        tmp = tmp.T  # for make it row vector again
        Wtmp = W_hid[i]
        tmp = np.matmul(tmp, W_hid[i])
        layer_activ(tmp, nHL)
        tmp = tmp.T  # for make it column vector
        H_hid[:, i] = tmp[:, 0]
    tmp = tmp.T  # for make it row vector again
    tmp = np.matmul(tmp, W_out)
    layer_activ(tmp, out_size)
    H_out[:] = tmp
    # error=cost_func_square()
    if (type):
        if (out_size == 1):
            print("-------------------------------")
            print("H_out[0,0]=",H_out[0,0])
            print("label= ",label)
            print("error squared= ",cost_func_square(H_out[0,0], label))
            return deriv_cost_mse(H_out[0,0], label)  # choose the cost function and her derivative
        else:
            pass  # for output layer which one have more than one neuron
            return 0
    else:
        if (out_size == 1):
            print(H_out[0,0])
            return H_out[0,0]  # choose the cost function and her derivative

        else:
            pass  # for output layer which one have more than one neuron
            return 0

# in backprop we use error value instead label value in forward, error returned from forward function
# added three more global variables in the end, for update weights of connections in network
def back_prop(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set, error, out_size, upd_W_in, upd_W_hid, upd_W_out, learning_rate=0.001):
    # error_new = []
    error_new=np.zeros((nHL,out_size))
    if (out_size == 1):
        for l in range(nHL):
            for r in range(out_size):
                # error_new[l,0] = np.matmul(W_out[l, :], error)
                # leftarg=W_out[l,:] #deletable
                #
                # error_val=W_out[l, :]*error
                # # error_new[l,0] = W_out[l, :]*error
                # leftarg=np.array([W_out[l,:]])
                # # rightarg=error
                # # tmp_err = np.matmul(W_out[l],error)
                # # tmp_err=np.matmul(leftarg,rightarg)
                # tmp_val=leftarg[0,l]*error
                # # error_new[l,0]=leftarg[0,l]*error
                valuer=W_out[l,0]*error
                error_new[l,0]=W_out[l,0]*error
                # error_new[l,0]=tmp_err[0,0]
                #-------------checker--------------------------------------
                tmp_upd_W_out=error*H_hid[l,(HL-1)]*H_out[r,0]*(1.0-H_out[r,0])
                tmp_error=error
                tmp_H_hid=H_hid[l,(HL-1)]
                tmp_H_out=H_out[r,0]
                tmp_H_out_2=(1.0-H_out[r,0])
                if (tmp_H_out_2==0):
                    tmp_H_out_2 = (1.0 - 0.999*H_out[r, 0])
                upd_tmp=error*H_hid[l,(HL-1)]*H_out[r,0]*tmp_H_out_2
                # print(upd_tmp)
                upd_W_out[l,r]=error*H_hid[l,(HL-1)]*H_out[r,0]*tmp_H_out_2
                #-------------checker--------------------------------------
                # upd_W_out[l,r]=error*H_hid[l,(HL-1)]*H_out[r,0]*(1.0-H_out[r,0]) # because of derivative of sigmoid
    else:
        pass #for future, if output signals will be more than one
    for L in range ((HL-2),-1,-1):
        error=np.copy(error_new)
        error_new = np.zeros((nHL, out_size))
        for l in range(nHL):
            for r in range(nHL):
                #----delete-------------
                tmpL=L+1
                H_hid_tmp=H_hid[l,L]
                #----delete-------------
                # error_new[l,0]=tmp_err[0,0]
                # upd_W_hid[(L+1),l,r]=error[r,0]*H_hid[l,L]*H_hid[r,(L+1)]*(1.0-(H_hid[r,(L+1)]))

                #---------------------checker---------------
                error_val=error[r,0]
                checker_tmp_W=W_hid[(L+1),[l]]
                s=checker_tmp_W
                #---------------------checker---------------
                tmp1=error[r,0]
                tmp2=H_hid[l,L]
                tmp3=(1.0-(H_hid[r,(L+1)]))
                tmp4=H_hid[r,(L+1)]
                tmp_resulter=tmp1*tmp2*tmp3*tmp4
                if(tmp3==0):
                    tmp3=(1.0-0.999*(H_hid[r,(L+1)]))
                tmp_resulter=error[r,0]*H_hid[l,L]*H_hid[r,(L+1)]*tmp3
                upd_W_hid[(L+1),l,r]=error[r,0]*H_hid[l,L]*H_hid[r,(L+1)]*tmp3
                #---------------------checker end---------------
                tmp_err=np.matmul(W_hid[(L+1),[l]],error)
                error_new[l,0]=tmp_err[0,0]
                # upd_W_hid[(L+1),l,r]=error[r,0]*H_hid[l,L]*H_hid[r,(L+1)]*(1.0-(H_hid[r,(L+1)]))

    error=np.copy(error_new) #error=error_new.copy()
    error_new=np.zeros((nHL,out_size))
    for l in range(nHL):
        for r in range(nHL):
            tmp_err = np.matmul(W_hid[0, [l]], error)
            error_new[l, 0] = tmp_err[0, 0]
            error_val=error[r,0]
            val2=H_in[l,0]
            val3=H_hid[r,0]
            val4=(1.0-(H_hid[r,0]))
            #---------checker---------
            if(H_hid[r,0]==1):
                tmp4=(1.0-0.999*(H_hid[r,0]))
            upd_W_hid[0, l, r] = error[r, 0] * H_in[l, 0] * H_hid[r, 0] * tmp4
            # ---------checker---------
            # value=error[r,0]*H_in[l,0]*H_hid[r,0]*(1.0-(H_hid[r,0]))
            # upd_W_hid[0,l,r]=error[r,0]*H_in[l,0]*H_hid[r,0]*(1.0-(H_hid[r,0]))

    error=np.copy(error_new) #error=error_new.copy()
    i = in_data_set.reshape(input_size, 1)  # just a data of one example, column vector
    for l in range(input_size):
        for r in range(nHL):
            error_val=error[r,0]
            #------------------checker---------------
            tmp_error=error[r,0]
            tmp2=i[l,0]
            tmp3=H_in[r,0]
            tmp4=(1.0-(H_in[r,0]))
            if(tmp4==0):
                tmp4=(1.0-0.999*(H_in[r,0]))
            tmp_result=tmp_error*tmp2*tmp3*tmp4
            # print("tmp result= ", tmp_result)
            upd_W_in[l,r]=error[r,0]*i[l,0]*H_in[r,0]*tmp4
            #------------------checker---------------
            # upd_W_in[l,r]=error[r,0]*i[l,0]*H_in[r,0]*(1.0-(H_in[r,0]))

    tmp=-learning_rate*upd_W_in
    W_in[:]=W_in[:]-learning_rate*upd_W_in
    W_hid[:]=W_hid[:]-learning_rate*upd_W_hid
    W_out[:]=W_out[:]-learning_rate*upd_W_out
    #add lerning rate
    # update weights

# transposed convlution ядро на разреженную матрицу раскладывается


labels = []
# parsers_out=parser(labels)
# in_data_set = np.array(parser(labels), dtype=float)  # for 1 d output, or scalar output !!! so the neurons in output layer=1!!!
in_data_set = np.array(parser(labels), dtype=np.longlong)  # for 1 d output, or scalar output !!! so the neurons in output layer=1!!!
count_of_examples, exmpl_vector_d_size = in_data_set.shape
input_size = exmpl_vector_d_size  # number of inputs "of data", dimension of vector i
out_size = 1  # equal the size of neurons in ouput layer, also depends on parser that you chose

# ----input layer (connection between input layer and first layer in hidden layer)----------
HL = 4  # number of hidden layers
nHL = 100  # number of neurons in one hidden layer
W_in = np.random.rand(input_size, nHL)  # weights
# W_in = np.zeros((input_size, nHL))  # weights VER 2 with zeros
H_in = np.zeros((nHL, 1))  # vector E R^(nHLx1)
# ----------------hidden layer----------------
W_hid = np.random.rand(HL, nHL, nHL)  # weights
# W_hid = np.zeros((HL, nHL, nHL))  # weights VER 2 with zeros
H_hid = np.zeros((nHL, HL))  # matrix ofr output values for each neuron in hidden layers layer
# ----------output layer----------
W_out = np.random.rand(nHL, out_size)  # weights
# W_out = np.zeros((nHL, out_size))  # weights VER 2 with zeros
H_out = np.zeros((out_size, 1))
# -------------------------------------------
learning_rate = 0.001  # just a constant value for a while, maybe add a "momentum" in the fьюча or would make a custom "changer" for it

iterations = 10 ** 7  # number of iterations
#----rounder-----------
# rounder=1
# W_in=np.round(W_in,rounder)
# W_hid=np.round(W_hid,rounder)
# W_out=np.round(W_out,rounder)

#----rounder-----


#----learn------------#
for iteration in range(iterations):
    if (not(iteration%500)):
        print("please in")
        c=input()
        print("100 iterrations passed")
    else:
        pass
    for exmpl in range(count_of_examples):
        print("number of example= ", exmpl)
        # upd_W_in=np.zeros((input_size,nHL),dtype=np.longlong)
        # upd_W_hid=np.zeros((HL,nHL,nHL),dtype=np.longlong)
        # upd_W_out=np.zeros((nHL,1),dtype=np.longlong)
        # checker=np.zeros((nHL,1),dtype=np.longlong)
        upd_W_in=np.zeros((input_size,nHL),dtype=float)
        upd_W_hid=np.zeros((HL,nHL,nHL),dtype=float)
        upd_W_out=np.zeros((nHL,1),dtype=float)
        checker=np.zeros((nHL,1),dtype=float)
        error = forward(input_size, HL, nHL, W_in, H_in, W_hid, H_hid, W_out, H_out, in_data_set[exmpl], labels[exmpl], out_size)
        back_prop(input_size,HL,nHL,W_in,H_in,W_hid,H_hid,W_out,H_out,in_data_set[exmpl],error,out_size,upd_W_in, upd_W_hid, upd_W_out,learning_rate)

print("da")
# print(H_in)


print("empr")
print("gd")
