import torch
import time
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import json
from transformers import AutoTokenizer, BertModel,  BertTokenizer
#from torchtext.vocab import GloVe
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from dataset_combined_choice_mask import *
from collate_choice_mask import *
from torch.nn.utils.rnn import pad_sequence
from Models.model_2 import Resnet,Atten_vid,Bert,Model_NDesc_Atten,Model_Desc,Lstm_vid,Model_NDesc2
from torch.cuda.amp import autocast, GradScaler
import wandb
import warnings
import pandas as pd
import argparse

warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    #model params
    parser.add_argument('--test_json', type=str)
    parser.add_argument('--test_frame', type=str)
    parser.add_argument('--desc_vocab',type=str)
    parser.add_argument('--model',type=str)
    args = parser.parse_args()

    word2idx = read_json(args.desc_vocab)
    test_json_data = read_json(args.test_json)
    test_data_desc= json_preprocess(test_json_data,True)
    test_src = Clevrer_Dataset(test_data_desc,args.test_frame,word2idx,True)


    test_dataloader = DataLoader(test_src, 
                                batch_size=64,
                                collate_fn =collate_fn,
                                shuffle=False,
                                num_workers=30,
                                pin_memory = True)

    resnet = Resnet(768)
    atten = Lstm_vid(768,768)
    bert = Bert()
    vocab_dim = len(word2idx)
    model = Model_NDesc2(resnet,atten,bert,vocab_dim,768)


    model.load_state_dict(torch.load(args.model))
    model = model.to(device)

    out = []
    with torch.no_grad():
        des_correct = 0
        des_total = 0
        ndes_correct = 0
        per_option = 0
        ndes_total = 0
        ndes_total_per_opt = 0
        threshold = 0.5
        predict=[]
        scene_idx = []
        question_id =[]
        choice_id = []
        i=0
        for batch in tqdm(test_dataloader):
            vid  = batch["video"].to(device)
            scene = batch["scene_index"]
            q_id = batch["question_id"]

            quest = batch["question"].to(device)
            quest_type = batch["question_type"]
            choice_mask = batch["choice_mask"].to(device)
            mask1 = [s == "descriptive" for s in quest_type]
            mask2 = [s != "descriptive" for s in quest_type]
            mask3 = [s for s in quest_type if s!="descriptive" ]
            mask4 = [s == "counterfactual" for s in mask3]
            des_answer = batch["des_answer"].to(device)
            des_answer = des_answer[mask1]
            des_answer = des_answer.squeeze(1)
            ndes_answer = batch["ndes_answer"].to(device)
            ndes_answer = ndes_answer[mask2]
            scene = scene[mask2]
            q_id = q_id[mask2]
            scene_idx.extend(scene.tolist())
            question_id.extend(q_id.tolist())
            ndes_answer = ndes_answer[mask4]
            choice_mask = choice_mask[mask2]
            choice_mask = choice_mask[mask4]
            choice_id.extend(choice_mask)
            out_ndes = model(vid,quest,quest_type)
            predict.extend(out_ndes.tolist())

        df_ndesc = pd.DataFrame(list(zip(scene_idx,question_id,predict)),columns = ['Scene','Q_id','Pred'])
        df_ndesc.to_json("Ndes.json")

# python evaluate_ndes.py --test_json Dataloader/test.json --test_frame Dataloader/test_64frames --desc_vocab word2idx.json --model resnet_bert_ndesc_best.pth 