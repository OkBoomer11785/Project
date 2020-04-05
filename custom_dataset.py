import os
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from kitti_utils import generate_depth_map


##Important variables here!
main_path = "kitti_data"


def readlines(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def read_image(path,img_num,side,img_tmpl):
    if side=='l':
        im = np.array(Image.open(path+'/image_02/data/'+img_tmpl.format(img_num)+'.jpg').convert('RGB'))
    else:
        im = np.array(Image.open(path+'/image_03/data/'+img_tmpl.format(img_num)+'.jpg').convert('RGB'))
    return im


def read_depth_data(depth_path,img_num,img_tmpl,cam):
    velodyne_path  = depth_path+'/{:010d}.bin'.format(img_num)
    depth_gt = generate_depth_map(os.path.join(main_path,depth_path.split('/')[1]), velodyne_path, cam)
    return depth_gt




class MainDataLoader(Dataset):
    def __init__(self,split_file_path,num_scales,img_tmpl):
        self.img_locations = readlines(split_file_path)
        self.img_tmpl = img_tmpl
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
    def __getitem__(self,idx):
        loc = self.img_locations[idx].split()
        path = os.path.join(main_path,loc[0]) #2011_09_26/2011_09_26_drive_0086_sync
        img_num = loc[1] #47
        side = loc[2] #l
        img = read_image(path,int(img_num),side,self.img_tmpl)
        if side=='l':
            img_other = read_image(path,int(img_num),'r',self.img_tmpl)
            cam = 2
        else:
            img_other = read_image(path,int(img_num),'l',self.img_tmpl)
            cam = 3
            
        depth_path = os.path.join(path,'velodyne_points/data')
        depth_gt = read_depth_data(depth_path,int(img_num),self.img_tmpl,cam)

        
if __name__ == '__main__':
    train_img_path = "splits/benchmark/train_files.txt"
    test_img_path = "splits/benchmark/test_files.txt"
    test_loader = MainDataLoader(train_img_path,num_scales=4,img_tmpl='{:010d}')
    test_loader[3]