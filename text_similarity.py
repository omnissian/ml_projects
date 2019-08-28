import numpy as np
import pandas as pd
import codecs
import scipy as scp
import re
import string

file_obj=codecs.open("D:\Mind\Coursera\MathAndPython\week2\sentences.txt",'r',encoding='windows-1252')

text=[]
print("type(text)")
print(type(text))
data = file_obj.readlines()
# for line in data: text.__add__(line.strip())
print("type(data)")
print(type(data))
print ("data[0]")
print (data[0])
print("type(data[0])")
print(type(data[0]))
print("\n------\n")

for n in data:
    text.append((str.lower(n)))
print ("data[0]")
print(data[0])
# print(text[0])
print("text")
print (text)
print("text[0]")
print (text[0])
print("hello")
