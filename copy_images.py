from PIL import Image
from matplotlib import image
from matplotlib import pyplot
import numpy as np
import os
from os import listdir
from numpy import asarray
import subprocess
#import PySimpleGUI as sg
import csv
import math

count = 0
filepath = "/home/enio/OIDv4_ToolKit/OID/Dataset/validation/"
new_filepath = "/home/enio/PycharmProjects/person_presence_project/Dataset/not_person/img"

for dir_name in listdir(filepath):
    file_list = listdir(filepath + dir_name + "/")
    file_cnt = len(file_list)
    status_cnt = 0
    print("\nStatus for " + dir_name + " folder:")
    for file_name in file_list:
        count += 1
        status_cnt += 1
        print("\r" + str(round((status_cnt/file_cnt)*100)) + "%", end="", flush=True)
        img = Image.open(filepath + dir_name + "/" + file_name)
        img = asarray(img)
        height = img.shape[0]
        width = img.shape[1]

        if height > width:
            h1 = math.floor((height - width)/2)
            h2 = h1 + width
            img = img[h1:h2, 0:width, ...]
        elif width > height:
            w1 = math.floor((width - height)/2)
            w2 = w1 + height
            img = img[0:height, w1:w2, ...]
        elif height == width:
            img = img[0:height, 0:width, ...]
        else:
            print("\nError in image: " + file_name + "\n")

        img = Image.fromarray(img)
        img.save(new_filepath + str(count) + ".png", format='PNG')
