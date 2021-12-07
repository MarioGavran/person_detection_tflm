from PIL import Image
from os import listdir

folder = ""
while folder not in ["person", "not_person"]:
    folder = input("Enter p/n for person or not_person: ")
    if folder == "p":
        folder = "person"
    elif folder == "n":
        folder = "not_person"
    else:
        print("Enter either 'p' or 'n' ")

size = ""
while size not in range(10, 200):
    size = input("Enter size 10-200: ")
    size = int(size)
    if size not in range(10, 200):
        print("Out of range")

print("Resizing images from '/" + folder + "/' to size " + str(size) + "x" + str(size) + " ...")
filepath = "/home/enio/PycharmProjects/person_presence_project/Dataset/" + folder + "/"
new_filepath = "/home/enio/PycharmProjects/person_presence_project/Dataset/" + folder + "_120x120/"

file_list = listdir(filepath)
file_count = len(file_list)
file_counter = 0
for filename in file_list:
    file_counter += 1
    img = Image.open(filepath + filename)
    img = img.resize((size, size))
    img.save(new_filepath + filename, format='PNG')
    print("\rStatus: " + str(round((file_counter/file_count) * 100)) + "%", end="", flush=True)
