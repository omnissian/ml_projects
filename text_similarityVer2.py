import numpy as np
import pandas as pd
import codecs
import scipy
import re
from scipy.spatial.distance import cdist
import string

file_obj=codecs.open("D:\Mind\Coursera\MathAndPython\week2\sentences.txt",'r',encoding='windows-1252')

data2d_sent_by_words=[]
text=[]
words=[]
matrix_of_sentences=[]
vector_of_sentence=[]
sentence=[]
dictionary=set()
# words = [[0 for j in range(m)] for i in range(n)]

for n in file_obj.readlines():
    # print("print (type(n) ", type(n))
    # print("print (n) ", n)
    # text.append(str.lower(n))
    tmp=re.split('[^a-z]', str.lower(n))
    sentence=[]
    # print(" print(type(tmp)) ", type(tmp))
    # print("print(tmp.__sizeof__())", tmp.__sizeof__())
    # print("print( tmp.__len__()) ",tmp.__len__())
    # print("print(tmp)", tmp)
    # print("print(tmp[3])", tmp[3])
    # print((tmp[3]=='')) # true
    for n in tmp:
        if (n!=''):
            # print (n)
            sentence.append(n)
            dictionary.add(n)
    data2d_sent_by_words.append(sentence)

#----------------
x=0
for n in data2d_sent_by_words:
    x+=1
    print (x, n)



#------------------
    # dictionary.add()
print("here we prin words")
matrix=[]
dictn=list(dictionary)
print ("print (type(dictn))", type(dictn))
print ("print (dictn[2])", dictn[2])
for n in data2d_sent_by_words:
    vector_of_sentence=[0]*len(dictn)
    for word in n:
        if (dictn.count(word)):
            vector_of_sentence[dictn.index(word)]=dictn.count(word)
    matrix.append(vector_of_sentence)

# print ("print (matrix)",matrix)

matrix_new=np.asarray(matrix)
print ("print (type(matrix))", type(matrix))
print ("print (type(matrix_new))", type(matrix_new))
print("matrix")
for n in matrix:
    print (n)
print("matrix_new")
print(matrix_new)
# for n in matrix:
#     print (n)

dist=[]
for row in matrix_new:
    dist.append(scipy.spatial.distance.cosine(matrix_new[0,:], row))
print("print(dists)", dist)
print ("print (type(dists))", type(dist))
dist_s=[]
dist_s=dist.copy()
dist_s.sort()
print("print(dist_s)", dist_s)
print("print(len(dist_s))", len(dist_s))
sentence_num=[]
sentence_num.append(dist.index([dist_s[1]]))
sentence_num.append(dist.index([dist_s[2]]))
print("print(sentence_num[1])")
print(sentence_num[0])
print("print(sentence_num[2])")
print(sentence_num[1])



# print("here we finished")
#
# # ads={1,4,3,4,1,1,1}
# # print("print(ads)", ads)
# print("print(len(dictionary))", len(dictionary))
#
# print("here we go")
# print(dictionary)
# print("print(data2d_sent_by_words[1][2])", data2d_sent_by_words[1][2])
# print("print(dictionary.__sizeof__())", dictionary.__sizeof__()) # 4196
# print("print(type(dictionary))", type(dictionary))
#



# re.split('[^a-z]', t)










