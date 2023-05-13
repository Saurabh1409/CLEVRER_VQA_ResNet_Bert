import json
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from natsort import natsorted
from transformers import BertTokenizer
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader
import warnings
import time

warnings.filterwarnings("ignore")

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer.save_pretrained("./Bert_Token/tokenizer/")
tokenizer = BertTokenizer.from_pretrained("./Bert_Token/tokenizer/")

transform = transforms.Compose([transforms.ToTensor(),
                                #transforms.Resize((128,128)),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

def json_preprocess(data, test = False):
    full_data = []
    for examples in data:
        for instance in examples['questions']:
            quest_desc= {}
            if instance["question_type"] == "descriptive":
                quest_desc['question_id'] = instance['question_id']
                quest_desc['question_type'] = instance['question_type']
                quest_desc['question'] = instance['question']
                quest_desc["choices"] = None
                quest_desc['video_filename'] = examples['video_filename']
                if test:
                    quest_desc['answer'] = ""
                else:
                    quest_desc['answer'] = instance['answer']
                full_data.append(quest_desc)
            else:
                quest_desc['question_id'] = instance['question_id']
                quest_desc['question_type'] = instance['question_type']
                quest_desc['question'] = instance['question']
                quest_desc["choices"] = instance['choices']
                quest_desc['video_filename'] = examples['video_filename']
                if test:
                    quest_desc['answer'] = ""
                else:
                    quest_desc['answer'] = None
                full_data.append(quest_desc)
    return full_data

def descriptive_vocab(data):
    words = set()
    word2idx = {}
    idx2word = {}
    counter = 0
    for examples in data:
        questions = examples['questions']
        for quest in questions:
            if quest["question_type"] == 'descriptive':
                if quest["answer"] not in words:
                    words.add(quest['answer'])
                    word2idx[quest['answer']] = counter
                    idx2word[counter] = quest['answer']
                    counter += 1
                else:
                    continue
    return word2idx , idx2word

def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data


class Clevrer_Dataset(Dataset):
    def __init__(self, data,frame_path,word2idx,test = False):
        self.data = data
        self.frame_path = frame_path
        self.word2idx = word2idx
        self.test = test
    def __len__(self):
        return len(self.data)

    def image_data(self,idx):
        new_data = self.data[idx]
        file_name = new_data['video_filename']
        out_path = self.frame_path+'/'+os.path.splitext(file_name)[0]
        frames_list = os.listdir(out_path)
        start = time.time()
        frames_list = natsorted(frames_list)
        img_data = []
        for frames in frames_list:
            start = time.time()
            frame = Image.open(out_path+'/'+frames)
            end = time.time()
            #print("reading Frame time : ",(end-start)/60)
            start = time.time()
            frame = transform(frame)
            end = time.time()
            #print("Frame TRANSFORM time : ",(end-start)/60)
            img_data.append(frame)
        start = time.time()
        img_data=torch.stack(img_data)
        end = time.time()
        #print("Frame stack time : ",(end-start)/60)
        return img_data

    def question_data(self, idx):
        sample = self.data[idx]
        quest_data = {}
        if sample["question_type"] == 'descriptive':
            quest_data["question_id"] = [sample["question_id"]]
            quest_data["question_type"] = sample["question_type"]
            quest_append = sample["question_type"] + " " + sample["question"]
            quest_data["question"] = [tokenizer(quest_append)['input_ids']]
            if self.test:
                quest_data["des_answer"] = [1035]
            else:
                quest_data["des_answer"] = [self.word2idx[sample["answer"]]]
            quest_data['video'] = self.image_data(idx)
            ans = torch.zeros(4,dtype=torch.long)
            quest_data["ndes_answer"] = ans
            return quest_data
        else:
            quest_data["question_id"] = [sample["question_id"]]
            quest_data["question_type"] = sample["question_type"]
            quest_append = sample["question_type"] + " " + sample["question"]
            #quest_data["question"] = [tokenizer(sample["question"])['input_ids']]
            ans = torch.zeros(4,dtype=torch.long)
            count = 0
            for i,choice in enumerate(sample["choices"]):
                quest_append = quest_append + " choice_"+str(i)+ " " + choice["choice"]
                if self.test:
                    continue
                else:
                    if choice["answer"]=="correct":
                        ans[i]=1
            quest_data["question"] = [tokenizer(quest_append)['input_ids']]
            quest_data["ndes_answer"] = ans
            quest_data["des_answer"] = [0]
            quest_data['video'] = self.image_data(idx)
            return quest_data

    def __getitem__(self, idx):
        quest_data = self.question_data(idx)
        return quest_data
