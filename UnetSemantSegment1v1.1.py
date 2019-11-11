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
path_valid_in="/storage/MazurM/Task1/validation/img/"
path_valid_out="/storage/MazurM/Task1/validation/mask_build/"
# im = Image.open("/storage/MazurM/Task1/ImgData/img/205_0_0.tiff")

# names_learn_in=os.listdir("/storage/MazurM/Task1/ImgData/img/")
names_learn_in=os.listdir(path_learn_in)

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
    # train_data.append((np.array(Image.open(path_learn_in+names_learn_in[i])),np.array(Image.open(path_learn_out+names_learn_in[i]))))
    # img_in=(np.array(Image.open(path_learn_in+names_learn_in[i])))
    # img_out=(np.array(Image.open(path_learn_out+names_learn_in[i])))
    img_in=Image.open(path_learn_in+names_learn_in[i])
    img_out=Image.open(path_learn_out+names_learn_in[i])
    train_data.append((np.array(img_in),np.array(img_out)))
    if(random.randint(1,100)>60):
        rotation=random.randint(1,3)
        train_data.append((img_in.rotate(rotation*90,expand=True),img_out.rotate(rotation*90,expand=True)))
        # learn_in.append(np.array(Image.open(path_learn_in + names_learn_in[i])))

names_valid_in=os.listdir(path_valid_in)
test_data=[]
for i in range(len(names_valid_in)):

    img_in=Image.open(path_valid_in+names_valid_in[i])
    img_out=Image.open(path_valid_out+names_valid_in[i])
    test_data.append((np.array(img_in),np.array(img_out)))
    if(random.randint(1,100)>50):
        rotation=random.randint(1,3)
        test_data.append((img_in.rotate((rotation*90),expand=True),img_out.rotate(rotation*90,expand=True)))

print("all is done")

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
