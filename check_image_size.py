from numpy import asarray
from PIL import Image
from os import listdir
from os.path import basename, dirname

filepath1 = "./Dataset/person/"
filepath2 = "./Dataset/not_person/"
for file_path in [filepath1, filepath2]:
    file_list = listdir(file_path)
    print("number of files in " + basename(dirname(file_path)) + " is: " + str(len(file_list)))
    for file_name in file_list:
        img = Image.open(file_path + file_name)
        img = asarray(img)
        height = img.shape[0]
        width = img.shape[1]
        if height == width:
            continue
        elif height > width:
            print(file_name)
            img = img[0:width, 0:width]
        elif width > height:
            print(file_name)
            img = img[0:height, 0:height]
        img = Image.fromarray(img)
        img.save(file_path + file_name, format='PNG')
