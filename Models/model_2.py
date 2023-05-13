import torch
import time
from PIL import Image
import os
from natsort import natsorted
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from torchvision.models import resnet34, ResNet18_Weights
import warnings

warnings.filterwarnings("ignore")

class Resnet(nn.Module):
    def __init__(self,out_dim):
        super(Resnet,self).__init__()
        self.out_dim = out_dim
        self.resnet = resnet34(pretrained=True)
        self.resnet.conv1 = torch.nn.Conv2d(3, 64, 
                                        kernel_size=(7, 7),
                                        stride=(2, 2), 
                                        padding=(3, 3), 
                                        bias=False)
        self.fc = nn.Linear(1000,self.out_dim)
        self.relu = nn.ReLU()
    def forward(self,x):
        b,f,c,h,w = x.shape
        x = x.reshape(-1,c,h,w)
        out = self.resnet(x)
        out = self.fc(out)
        out = self.relu(out)
        out = out.reshape(b,f,-1)
        return out

class Lstm_vid(nn.Module):
    def __init__(self,in_dim,hid_dim,n_layers=1):
        super(Lstm_vid,self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(in_dim,hid_dim,
                            num_layers = n_layers,
                            batch_first = True)
    def forward(self,x):
        out, (h_n, c_n) = self.lstm(x)
        hidden = h_n[-1]
        return out,hidden

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        #self.fc = nn.Linear(768,out_dim)
    def forward(self, input_id):
        outputs = self.bert(input_id)
        enc_outs = outputs['last_hidden_state']
        pooled_output = outputs['pooler_output']
        #pooled_output = self.fc(pooled_output)
        return pooled_output

class Model_Desc(nn.Module):
    def __init__(self,resnet,lstm,bert,vocab_dim,out_dim):
        super(Model_Desc,self).__init__()
        self.lstm = lstm
        self.resnet = resnet
        self.bert = bert
        self.vid_quest_fc = nn.Linear(2*out_dim,out_dim)
        self.fc1_des = nn.Linear(out_dim,100)
        self.fc2_des = nn.Linear(100,vocab_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self,vid,question,quest_type):
        resnet_out = self.resnet(vid)
        _,vid_emb = self.lstm(resnet_out)
        quest_emb = self.bert(question)
        vq_emb = torch.cat((vid_emb,quest_emb),1)
        vq_emb = self.vid_quest_fc(vq_emb)
        vq_emb = self.dropout(vq_emb)
        mask1 = [s == "descriptive" for s in quest_type]
        vq_emb_des = vq_emb[mask1]
        vq_des = self.fc1_des(vq_emb_des)
        vq_des = self.dropout(vq_des)
        out_des = self.fc2_des(vq_des)
        return out_des 

class Model_NDesc2(nn.Module):
    def __init__(self,resnet,lstm,bert,vocab_dim,out_dim):
        super(Model_NDesc2,self).__init__()
        self.lstm = lstm
        self.resnet = resnet
        self.bert = bert
        self.vid_quest_fc = nn.Linear(2*out_dim,out_dim)
        self.fc1_ndes = nn.Linear(out_dim,100)
        self.fc2_ndes = nn.Linear(100,4)
        self.dropout3 = nn.Dropout(0.2)
        self.sig = nn.Sigmoid()

    def forward(self,vid,question,quest_type):
        resnet_out = self.resnet(vid)
        _,vid_emb = self.lstm(resnet_out)
        quest_emb = self.bert(question)
        vq_emb = torch.cat((vid_emb,quest_emb),1)
        vq_emb = self.vid_quest_fc(vq_emb)
        vq_emb = self.dropout3(vq_emb)
        mask2 = [s != "descriptive" for s in quest_type]
        vq_emb_ndes = vq_emb[mask2]
        vq_ndes = self.fc1_ndes(vq_emb_ndes)
        vq_ndes = self.dropout3(vq_ndes)
        out_ndes = self.fc2_ndes(vq_ndes)
        out_ndes = self.sig(out_ndes)
        return out_ndes
