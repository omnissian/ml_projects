from PIL import Image
import matplotlib as plt # pyplot!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision as tv
import os
import random
random.seed(777)


path_learn_in="/storage/MazurM/Task1/ImgData/img/"
path_learn_out="/storage/MazurM/Task1/ImgData/mask_build/"
im = Image.open("/storage/MazurM/Task1/ImgData/img/205_0_0.tiff")

names_learn_in=os.listdir("/storage/MazurM/Task1/ImgData/img/")
learn_in=[]
learn_out=[]

# print(names_learn_in[2])
# print(type(names_learn_in[2]))
# path_file=path_learn_in+names_learn_in[2]
# print(path_file)
# print(type(path_learn_in))
train_data=[]

tuple1=tuple()
for i in range(len(names_learn_in)):
    # learn_in.append(np.array(Image.open(path_learn_in+names_learn_in[i])))
    # learn_out.append(np.array(Image.open(path_learn_out+names_learn_in[i])))
    # train_data.append((learn_in[i],learn_out[i]))
    train_data.append((np.array(Image.open(path_learn_in+names_learn_in[i])),np.array(Image.open(path_learn_out+names_learn_in[i]))))
    if(random.randint(1,100)>60):
        rotation=random.randint(0,3)
        learn_in.append(np.array(Image.open(path_learn_in + names_learn_in[i])))


# for i in range(len(names_learn_in)):
#     with open(path_learn+names_learn_in[i]) as in_img:
#
#         wtf=Image.open(in_img)
#         wtf2=np.array(wtf)
#         learn_in.append(np.array(Image.open(in_img)))
# print("delta")



print(type(im))
x=np.array(im)
print(x)
print(type(x))
print(x.shape)
plt.imshow(x)
plt.show()

# im.show()
# x.show()
# im.rotate(45).show()
