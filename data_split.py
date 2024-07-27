import os
import random
import shutil
from itertools import islice

output_folder_path = "D:/output_data"
input_folder_path = "D:/data_collection"
splitratio = {'train': 0.7, 'test': 0.1, 'validation': 0.2}
classes =  ['real','fake']
try:
    shutil.rmtree(output_folder_path)
except OSError as e:
    os.mkdir(output_folder_path)

os.makedirs(f"{output_folder_path}/train/images",exist_ok=True)
os.makedirs(f"{output_folder_path}/train/labels",exist_ok=True)
os.makedirs(f"{output_folder_path}/val/images",exist_ok=True)
os.makedirs(f"{output_folder_path}/val/label",exist_ok=True)
os.makedirs(f"{output_folder_path}/test/images",exist_ok=True)
os.makedirs(f"{output_folder_path}/test/labels",exist_ok=True)

# get the names
list_names = os.listdir()
print(list_names)
unique_names = []
for filename in list_names:
    filename = filename.split('.')
    if filename[0] not in unique_names:
        unique_names.append(filename[0])

# shuffle
random.shuffle(unique_names)
# find the number of images for each folder
lentrain  = int(len(unique_names) * splitratio['train'])
lentest  = int(len(unique_names) * splitratio['test'])
lenval  = int(len(unique_names) * splitratio['validation'])

# split the list
lengthtosplit = [lentrain, lentest, lenval]
input = iter(unique_names)
output = [list(islice(input, elem))for elem in lengthtosplit]

# copy the files
sequence =['train','test','validation']
for i,out in enumerate(output):
    for filename in out:
        shutil.copy(f'{input_folder_path}/{filename}.jpg', f'{output_folder_path}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{input_folder_path}/{filename}.txt', f'{output_folder_path}/{sequence[i]}/labels/{filename}.text')

# create the yaml file for the yolo model
dataYaml = f'path: ../Data\n\
train: ../train/images\n\
test: ../test/images\n\
val: ../val/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'


f = open(f"{output_folder_path}/data.yaml", 'a') #same path to be used above also in dataYaml
f.write(dataYaml)
f.close()
