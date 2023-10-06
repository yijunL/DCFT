import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import json

from proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, RobertaModel, RobertaTokenizer, RobertaForSequenceClassification
from sentence_encoder import BERTSentenceEncoder, BERTPAIRSentenceEncoder, RobertaSentenceEncoder, RobertaPAIRSentenceEncoder
import torch.utils.data as data
import random
from tqdm import tqdm



class FewShotTestREFramework:

    def __init__(self, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''

        self.test_data_loader = test_data_loader


    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def test(self,
             model,
             B, N, K, Q,
             eval_iter,
             na_rate=0,
             ckpt=None):
        '''
        model: a FewShotREModel instance
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")

        model.eval()
        if ckpt is None:
            print("Use val dataset")
            eval_dataset = self.val_data_loader
        else:
            print("Use test dataset")
            if ckpt != 'none':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_sample = 0.0
        #preds = []
        lists = []
        with torch.no_grad():
            for i in tqdm(list(range(10000))):
                batch, relation = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in batch:
                        batch[k] = batch[k].cuda()
                    for r in relation:
                        relation[r] = relation[r].cuda()
                logits, pred , _ = model(batch, relation, N, K, 1)

                newpred = pred.cpu()
                ls = newpred.numpy().tolist()
                for i in ls:
                    lists.append(i)

        with open("pred-"+str(N)+"-"+str(K)+".json", "w") as f: #############
            f.write(str(lists))
        return lists


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root, single=False):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)

        self.json_data = json.load(open(path))

        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.istrain = True
        self.single = single

    def __getraw__(self, item):
        word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask = self.encoder.tokenize(item['tokens'],
                                                               item['h'][2][0],
                                                               item['t'][2][0],
                                                               item['h'][0],
                                                               item['t'][0],
                                                               self.istrain)
        return word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask

    def __additem__(self, d, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['pos3'].append(pos3)
        d['pos4'].append(pos4)
        d['pos5'].append(pos5)
        d['pos6'].append(pos6)
        d['pos7'].append(pos7)
        d['pos8'].append(pos8)
        d['mask'].append(mask)

    def __getitem__(self, index):
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                       'pos8': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                     'pos8': [], 'mask': []}

        newindex = index * 1
        for i in range(newindex, newindex + 1):######################
            data = self.json_data[i]
            test = data["meta_test"]
            word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask = self.__getraw__(test)
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            pos3 = torch.tensor(pos3).long()
            pos4 = torch.tensor(pos4).long()
            pos5 = torch.tensor(pos5).long()
            pos6 = torch.tensor(pos6).long()
            pos7 = torch.tensor(pos7).long()
            pos8 = torch.tensor(pos8).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask)
            trains = data["meta_train"]
            for train in trains:
                # 外层大循环
                support = []
                for trainsample in train:
                    # 内层实例
                    word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask = self.__getraw__(trainsample)
                    word = torch.tensor(word).long()
                    pos1 = torch.tensor(pos1).long()
                    pos2 = torch.tensor(pos2).long()
                    pos3 = torch.tensor(pos3).long()
                    pos4 = torch.tensor(pos4).long()
                    pos5 = torch.tensor(pos5).long()
                    pos6 = torch.tensor(pos6).long()
                    pos7 = torch.tensor(pos7).long()
                    pos8 = torch.tensor(pos8).long()
                    mask = torch.tensor(mask).long()
                    self.__additem__(support_set, word, pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8, mask)

        return support_set, query_set

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                     'pos8': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'pos3': [], 'pos4': [], 'pos5': [], 'pos6': [], 'pos7': [],
                   'pos8': [], 'mask': []}
    batch_label = []
    support_sets, query_sets= zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
    for i in range(len(query_sets)):
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        # batch_label += query_labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    # batch_label = torch.tensor(batch_label)
    return batch_support, batch_query


def get_loader(name, encoder, N, K, Q, batch_size,
               num_workers=0, collate_fn=collate_fn, na_rate=0, root='../data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)



def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,###################
                        help='K shot')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batch size')
    parser.add_argument('--model', default='proto',
                        help='model name')
    parser.add_argument('--encoder', default='bert',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=128, type=int,
                        help='max length')
    parser.add_argument('--load_ckpt', default='checkpoint/proto-bert-train_wiki-val_pubmed-5-1-adv_pubmed_unsupervised-catentity-51R-8.pth.tar',
                        help='load ckpt')

    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    opt = parser.parse_args()
    
    test_file = "test_pubmed_input-"+str(opt.N)+"-"+str(opt.K)

    N = opt.N
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    encoder_name = opt.encoder
    max_length = opt.max_length
    pretrain_ckpt = "../pretrain/bert-base-uncased"
    sentence_encoder = BERTSentenceEncoder(
        pretrain_ckpt,
        max_length)
    test_data_loader = get_loader(test_file, sentence_encoder,
                N=N, K=K, Q=Q, batch_size=batch_size)
    model = Proto(sentence_encoder)
    if torch.cuda.is_available():
        model.cuda()
    ckpt = 'checkpoint/proto-bert-train_wiki-val_pubmed-'+str(opt.N)+'-'+str(opt.K)+'-adv_pubmed_unsupervised-catentity-51R-8.pth.tar'
    framework = FewShotTestREFramework(test_data_loader)
    result = framework.test(model, batch_size, N, K, Q, 1, 0, ckpt=ckpt)
    print(len(result))


if __name__ == "__main__":
    main()
