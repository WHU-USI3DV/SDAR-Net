import os
import numpy as np
import torch
import cv2
import torch.utils.data as dataf
dtype = 'float32'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.set_default_tensor_type(torch.FloatTensor)

from tqdm import tqdm

def get_data(batch_size=2,path="/data/UnderwaterDatasets/UIEB-new/train/", img_size=256):
    training_x=[]
    training_y=[]
    path_list = os.listdir(path+"input/")
    for item in tqdm(path_list):
        imgx= cv2.imread(path+"input/"+item)
        imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
        imgx=cv2.resize(imgx,(img_size,img_size))
        imgy= cv2.imread(path+"target/"+item)
        imgy = cv2.cvtColor(imgy, cv2.COLOR_BGR2RGB)
        imgy=cv2.resize(imgy,(img_size,img_size))
        training_x.append(imgx)
        training_y.append(imgy)
    X_train = np.array(training_x)
    X_train= torch.from_numpy(X_train.astype(dtype))
    X_train=X_train.permute(0,3,1,2)
    X_train=X_train/255.0
    print(X_train.shape)
    Y_train = np.array(training_y)
    Y_train= torch.from_numpy(Y_train.astype(dtype))
    Y_train=Y_train.permute(0,3,1,2)
    Y_train=Y_train/255.0
    print(Y_train.shape)
    train_dataset = dataf.TensorDataset(X_train,Y_train)
    train_loader = dataf.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=4)
    return train_loader

