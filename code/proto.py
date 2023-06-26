import random
import sys
sys.path.append('..')
import torch
from torch import autograd, optim, nn
import framework
from torch.autograd import Variable
from torch.nn import functional as F
import math
import random
from sklearn import metrics
def softplus(x):
    return torch.nn.functional.softplus(x)
class Proto(framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False):
        framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout(0.1) #########################Drop
        self.hidden_size = int(3*768)
        hidden_size = int(3*768)
        self.gamma = torch.nn.Parameter(torch.ones(1, hidden_size)*0.3)
        self.beta = torch.nn.Parameter(torch.ones(1, hidden_size)*0.5)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.activate = nn.ReLU()
        
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4)
        self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=1)
        self.weight_layer = nn.Linear(hidden_size,1)
        self.softmax = nn.Softmax()

    def feature_wise(self, support, query):
        
        rand_gamma = (1 + torch.randn(1, self.hidden_size, dtype=self.gamma.dtype, device=self.gamma.device)*softplus(self.gamma))
        rand_beta = (torch.randn(1, self.hidden_size, dtype=self.beta.dtype, device=self.beta.device)*softplus(self.beta))
        
        gamma = rand_gamma.expand_as(support)
        beta = rand_beta.expand_as(support)
        ada_support = support*gamma+beta
        
        gamma = rand_gamma.expand_as(query)
        beta = rand_beta.expand_as(query)
        ada_query = query*gamma+beta
        
        return ada_support, ada_query
    

    def __dist__(self, x, y, dim):
        simcos = torch.cosine_similarity(y, x, dim)

        return simcos

    def __batch_dist__(self, S, Q, f=None):

        if f is None:
            return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)  #[B N D]  [B totalQ D]
        else:
            return torch.cosine_similarity(Q.unsqueeze(2), S.unsqueeze(1), 3)


    def forward(self, support, query, N, K, total_Q, label=None, target=None):

        recon_loss = None

        support_emb, Sout, Srela, SCLS = self.sentence_encoder(support)  # (B * N * K, D), where D is the hidden size
        query_emb, Qout, Qrela, QCLS = self.sentence_encoder(query)  # (B * total_Q, D)
        
        if(target is not None):
            target_emb, Tout, Trela, TCLS = self.sentence_encoder(target)  # [10, D]
            target = self.drop(target_emb)
        # Drop
        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)

        if(target is not None and random.random()>0.5):
            support, query = self.feature_wise(support, query)
            recon_loss = (F.mse_loss(support.mean(),target.mean())+0.001*F.mse_loss(support.std(),target.std()))/2

        if(K>1):
            coa_support = support.view(N,K,hidden_size)
            coa_support = self.transformer_encoder2(coa_support).view(-1,hidden_size)
            support_weight = self.weight_layer(support)
            support_weight = self.softmax(support_weight.view(N,K)).view(-1,1)
            raw_support = support_weight*support
            raw_support = raw_support.view(N,K,hidden_size).sum(1)
            
        else:  #(N-way-1-shot)
            raw_support = support.view(N,hidden_size)
        
        querys = query
        total_logits = None
        total_pred = None
        for i in range(total_Q):
            query = querys[i].view(1,hidden_size)
            
            feats = self.transformer_encoder(torch.cat([raw_support,query],dim=0).view(1,N+1,hidden_size)).view(-1,hidden_size)
            aug_support,aug_query = feats[:N], feats[N:]

            support =raw_support+self.activate(aug_support)
            query = query+self.activate(aug_query)

            support = support.view(-1, N, hidden_size)  # (B, N, K, D)
            query = query.view(-1, 1, hidden_size)  # (B, total_Q, D)

            logits = self.__batch_dist__(support, query, None)  # (B 1 N D)  (B totalQ 1 D) (B, total_Q, N)
            logits = logits / 0.5
            _, pred = torch.max(logits.view(-1, N), 1)
            if(total_logits is None):
                total_logits = logits
                total_pred = pred
            else:
                total_logits = torch.cat([total_logits,logits],dim=0)
                total_pred = torch.cat([total_pred,pred],dim=0)
        
        return total_logits, total_pred, recon_loss


