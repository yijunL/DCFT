import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import os
from torch import optim
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('../pretrain/bert-base-uncased/vocab.txt')
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity

        if os.path.exists('stict.npy'):
            self.mlmdict = np.load('stict.npy', allow_pickle=True).item()
        else:
            self.mlmdict = {}

    def forward(self, inputs):

        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        tensor_range = torch.arange(inputs['word'].size()[0])
        # CLS = outputs[1]
        # print(outputs[0].shape)
        h_state = outputs[0][tensor_range, inputs["pos1"]]
        h2_state = outputs[0][tensor_range, inputs["pos2"]]
        t_state = outputs[0][tensor_range, inputs["pos3"]]
        t2_state = outputs[0][tensor_range, inputs["pos4"]]
#             e1 = outputs[0][tensor_range, inputs["pos5"]]
#             e2 = outputs[0][tensor_range, inputs["pos6"]]
        rela = inputs["pos7"]

#             state = torch.cat((h_state + e1, t_state + e2, h2_state-t2_state), -1)  #
        state = torch.cat((h_state, t_state, h2_state-t2_state), -1)  #

        return state, outputs[0], rela, outputs[1]
        # return  outputs[1]

    def tokenize(self, raw_tokens, pos_head, pos_tail, Ehead, Etail, istrain=True):

        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        pos3_in_index = 1
        pos4_in_index = 1
        pos5_in_index = 1
        pos6_in_index = 1
        pos7_in_index = 1
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[E1]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[/E1]')
                pos2_in_index = len(tokens)
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[E2]')
                pos3_in_index = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[/E2]')
                pos4_in_index = len(tokens)
            cur_pos += 1

#         pos7_in_index = len(tokens)

#         E1 = ['\"']+[raw_tokens[t] for t in pos_head]+['\"']
#         E2 = ['\"']+[raw_tokens[t] for t in pos_tail]+['\"']
#         zhishi1 = 'means'   # indicate    :   means :
#         zhishi2 = 'and'

#         for token in E1:
#             tokens += self.tokenizer.tokenize(token)
#         tokens+=self.tokenizer.tokenize(zhishi1)

#         tokens.append('[unused5]')
#         pos5_in_index = len(tokens)

#         tokens += self.tokenizer.tokenize(',')

#         for token in E2:
#             tokens += self.tokenizer.tokenize(token)
#         tokens+=self.tokenizer.tokenize(zhishi1)
#         # tokens += Eheadtoken  ##############
#         tokens.append('[unused6]')
#         pos6_in_index = len(tokens)

#         pos8_in_index =  1

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1

        
        pos5_in_index = 0
        pos6_in_index = 0
        pos7_in_index = 0
        pos8_in_index = 0
        
        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        pos3_in_index = min(self.max_length, pos3_in_index)
        pos4_in_index = min(self.max_length, pos4_in_index)
        pos5_in_index = min(self.max_length, pos5_in_index)
        pos6_in_index = min(self.max_length, pos6_in_index)
        pos7_in_index = min(self.max_length, pos7_in_index)
        pos8_in_index = min(self.max_length, pos8_in_index)

#         ansid = [pos1_in_index-1,pos3_in_index-1,pos2_in_index-1,pos4_in_index-1]
#         posid = sorted([pos4_in_index-1,pos3_in_index-1,pos2_in_index-1,pos1_in_index-1] ,reverse=True)
#         finalid = 0
#         for oneid in posid:
#             if '[unused' in tokens[oneid]:
#                 finalid = oneid
#                 break
#         for id in range(4):
#             if not '[unused' in tokens[ansid[id]]:
#                 ansid[id] = finalid

#         return indexed_tokens, ansid[0], ansid[1], ansid[2], ansid[3], pos5_in_index-1, pos6_in_index-1, \
#                pos7_in_index-1, pos8_in_index-1, mask
        return indexed_tokens, pos1_in_index-1,pos3_in_index-1,pos2_in_index-1,pos4_in_index-1, pos5_in_index-1, pos6_in_index-1, \
               pos7_in_index-1, pos8_in_index-1, mask


class BERTPAIRSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length): 
        nn.Module.__init__(self)
        self.bert = BertForSequenceClassification.from_pretrained(
                pretrain_path,
                num_labels=2)
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, inputs):
        x = self.bert(inputs['word'], token_type_ids=inputs['seg'], attention_mask=inputs['mask'])[0]
        return x
    
    def tokenize(self, raw_tokens, pos_head, pos_tail):
        # token -> index
        # tokens = ['[CLS]']
        tokens = []
        cur_pos = 0
        pos1_in_index = 0
        pos2_in_index = 0
        for token in raw_tokens:
            token = token.lower()
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)
            tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        
        return indexed_tokens