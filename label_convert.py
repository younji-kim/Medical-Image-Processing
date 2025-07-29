
import os
import cv2
import matplotlib.pyplot as plt

image_list=[]
file_name_list=[]
def search(dirname):
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        file_name_list.append(filename)
        #print(full_filename)
        image_list.append(full_filename)
search("./data_annotated/CT+MRI/val_json_36/MRI/SegmentationClassPNG")

output_directory = "./data_annotated/CT+MRI/val_json_36/MRI/SegmentationClassPNG3/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

for i in range(0,len(image_list)):
    iamge = cv2.imread(image_list[i])
    image_gray = cv2.cvtColor(iamge, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray/255
    output_path = os.path.join(output_directory, file_name_list[i])
    cv2.imwrite(output_path, image_gray)