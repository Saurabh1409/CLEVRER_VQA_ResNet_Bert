import torch
import time
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

def collate_fn(batch):
    #import pdb; pdb.set_trace()
    des_ans = [torch.LongTensor(s['des_answer']) for s in batch]
    ndes_ans = [torch.LongTensor(s['ndes_answer']) for s in batch]
    mask1 = ()
    q_type = [s['question_type'] for s in batch]
    q_id = [torch.LongTensor(s['question_id']) for s in batch]
    vid = [s['video'] for s in batch]
    sequences = [s['question'][0] for s in batch]
    max_length = max(len(s) for s in sequences)
    padded_sequences = [torch.nn.functional.pad(torch.tensor(s), 
                        (0, max_length - len(s))) for s in sequences]

    return {'video':torch.stack(vid),
            'question': torch.stack(padded_sequences),
            'des_answer': torch.stack(des_ans),
            'ndes_answer': torch.stack(ndes_ans),
            'question_type': q_type,
            'question_id':torch.stack(q_id)}