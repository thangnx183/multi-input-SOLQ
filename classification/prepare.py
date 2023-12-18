import os
import random
from tqdm import tqdm
import shutil
import glob


data_path = '../coco_data/dent/'

damage = glob.glob(data_path+'postive_crop/*')
# crack_train = damage[:15200]
no_damage = glob.glob(data_path+'negative_crop/*')

os.system('rm -rf ' + data_path + '20231208/train/')
os.system('rm -rf ' + data_path + '20231208/val/')

os.makedirs(data_path+'20231208/train/', exist_ok=True)
os.makedirs(data_path+'20231208/train/1', exist_ok=True)
os.makedirs(data_path+'20231208/train/0', exist_ok=True)

os.makedirs(data_path+'20231208/val/', exist_ok=True)
os.makedirs(data_path+'20231208/val/1', exist_ok=True)
os.makedirs(data_path+'20231208/val/0', exist_ok=True)


for f in tqdm(damage[:int(0.85*len(damage))]):
    shutil.copy(f, data_path+'20231208/train/0/' + f.split('/')[-1])

# for f in tqdm(glob.glob(data_path+'crop_crack/test/0/*')):
#     shutil.copy(f, data_path+'20231208/train/0/' + f.split('/')[-1])

for f in tqdm(no_damage[:int(0.85*len(no_damage))]):
    shutil.copy(f, data_path+'20231208/train/1/' + f.split('/')[-1])


for f in tqdm(damage[int(0.85*len(damage)):]):
    shutil.copy(f, data_path+'20231208/val/0/' + f.split('/')[-1])
    
for f in tqdm(no_damage[int(0.85*len(no_damage)):]):
    shutil.copy(f, data_path+'20231208/val/1/' + f.split('/')[-1])
