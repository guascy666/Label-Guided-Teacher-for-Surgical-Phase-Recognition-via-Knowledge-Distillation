import torch
import torch.nn as nn
import numpy as np

"""
以mask_length为宽度的条带Mask
"""
def get_mask(max_lenth,mask_lenth):
    #mask_lenth表示可以接收到多少前面moment的个数
    n=max_lenth
    mask=torch.ones(n,n)
    mask1=torch.ones(n,n)
    mask=torch.triu(mask,diagonal=-1*(mask_lenth-1))
    mask1=torch.triu(mask1,diagonal=1)
    return mask-mask1

"""
QKV注意力计算
"""
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, n_heads,mask_lenth,type):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_heads = n_heads
        self.mask_lenth=mask_lenth
        self.type=type

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q=1, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        mask_lenth=self.mask_lenth
        if mask_lenth>0:#-1代表没有mask
            mask = get_mask(scores.size(2), mask_lenth)
            mask = mask.cuda()
            scores = scores.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        elif mask_lenth==-2:#以前全部信息
            mask = get_mask(scores.size(2), scores.size(2))
            mask = mask.cuda()
            scores = scores.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


"""
多头注意力
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, len_q, len_k,mask_lenth,type='past'):
        super(MultiHeadAttention, self).__init__()

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ScaledDotProductAttention = ScaledDotProductAttention(self.d_k, n_heads,mask_lenth,type)
        self.len_q = len_q
        self.len_k = len_k
        self.mask_lenth=mask_lenth

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        #print("Q.SIZE",Q.size())
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]

        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context= self.ScaledDotProductAttention(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


"""
FeedForward
"""
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.d_model = d_model

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)


"""
Transformer Encoder 子结构
"""
class Encoder(nn.Module):
    def __init__(self,mask_lenth,d_model=512,d_ff=2048, d_k=64, d_v=64, n_heads=8, len_q=1):
        super(Encoder,self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q, mask_lenth)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self,enc_inputs):
        enc_outputs= self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


"""
Transformer Decoder 子结构
"""
class Decoder(nn.Module):
    def __init__(self,mask_lenth,d_model=512,d_ff=2048, d_k=64, d_v=64, n_heads=8, len_q=1):
        super(Decoder,self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, 1, len_q, mask_lenth)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
    def forward(self,enc_inputsq,enc_inputs2):
        enc_outputs= self.enc_self_attn(enc_inputsq, enc_inputs2, enc_inputs2)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs

"""
余弦位置编码
"""
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=10000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        

        self.register_buffer('pe', pe)

    def forward(self, x):
        pos = self.pe[: ,:x.size(1), :] + x
        return pos



"""
Label编码
"""
class Label_encoder(nn.Module):
    def __init__(self,dim,c=7):
        super(Label_encoder,self).__init__()
        self.dim=dim
        self.c=c

    def forward(self,inputs_label):
        length=inputs_label.size(0)
        encoding = torch.zeros((1,length,self.dim))
        segment_size = self.dim // self.c
        for i in range(length):
            phase=inputs_label[i]
            start = phase  * segment_size
            if phase==6:
                encoding[0,i,start:] = 1
            else:
                end = start + segment_size
                encoding[0,i,start:end] = 1
        return encoding.cuda()


# 降维TCN+NLP
class Down_TCN(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(Down_TCN,self).__init__()
        self.lin_down=nn.Linear(input_dim, output_dim, bias=False)
        self.tanh=nn.Tanh()
        # self.norm = nn.LayerNorm(output_dim)
        
    def forward(self,input):
        out1=self.lin_down(input)
        out1=self.tanh(out1)
        return out1


"""
Transformer最基本的Block for teacher
"""
# down and up path trans blocks
class Block_label(nn.Module):
    def __init__(self,input_dim,output_dim,mask_length):
        super(Block_label,self).__init__()
        self.Label_PE=FixedPositionalEncoding(embedding_dim=input_dim)
        self.enc_spa=Encoder(mask_lenth=mask_length,d_model=input_dim,d_ff=4*input_dim,d_k=int(input_dim/8),d_v=int(input_dim/8),n_heads=8)
        self.enc_label=Encoder(mask_lenth=mask_length,d_model=input_dim,d_ff=4*input_dim,d_k=int(input_dim/8),d_v=int(input_dim/8),n_heads=8)
        self.dec=Decoder(mask_lenth=mask_length,d_model=input_dim,d_ff=4*input_dim,d_k=int(input_dim/8),d_v=int(input_dim/8),n_heads=8)
        self.middle=Down_TCN(input_dim,output_dim)
        self.label_encoder=Label_encoder(dim=input_dim)

    def forward(self,spa,label):
        label_encoding=self.enc_label(self.Label_PE(self.label_encoder(label)))
        spa_out=self.enc_spa(spa)
        # dec_output=self.dec(spa_out,label_encoding)
        dec_output=self.dec(spa_out,label_encoding)
        out=self.middle(dec_output)

        return out


"""
用于Stage2和3 没有Label 的Transformer最基本的Block for student
"""
class Block_spa(nn.Module):
    def __init__(self,input_dim,output_dim,mask_length,n_layers):
        super(Block_spa,self).__init__()
        self.spa_PE=FixedPositionalEncoding(embedding_dim=input_dim)
        self.middle = Down_TCN(input_dim, output_dim)

        self.stages=nn.ModuleList([
            Encoder(mask_lenth=mask_length, d_model=input_dim, d_ff=4 * input_dim, d_k=int(input_dim / 8),
                               d_v=int(input_dim / 8), n_heads=8)
            for i in range(n_layers)
        ])

    def forward(self,input):
        #spa=self.spa_middle_in(self.spa_PE(self.lin_spa(spa)))
        out=input
        for layer in self.stages:
            out=layer(out)
        out=self.middle(out)
        return out

import math
"""
Teacher 网络 Encoder 部分
"""
# class Down_Encoder_Label(nn.Module):
#     def __init__(self,mask_length,dmodel,downsample_rate,active_tanh):
#         super(Down_Encoder_Label,self).__init__()
#         self.dim1=int(dmodel/downsample_rate)
#         self.dim2=int(self.dim1/downsample_rate)
#         self.block1 = Block_label(dmodel, self.dim1, mask_length)
#         self.block2 = Block_label(self.dim1, self.dim2, mask_length)
#         self.lin_down=nn.Linear(self.dim2,32)
#         self.tanh=nn.Tanh()
#         self.active_tanh=active_tanh

#     def forward(self,spa,label):
#         out1=self.block1(spa,label)
#         out2=self.block2(out1,label)
#         out3=self.lin_down(out2)
#         if self.active_tanh:
#             out3=self.tanh(out3)
#         return out3

class Down_Encoder_Label(nn.Module):
    def __init__(self,mask_length,dmodel,downsample_rate,active_tanh):
        super(Down_Encoder_Label,self).__init__()

        N=int(math.log(int(dmodel/64), int(downsample_rate)))

        self.stages=nn.ModuleList([
            Block_label(int(dmodel/int(math.pow(downsample_rate,i))),int(dmodel/int(math.pow(downsample_rate,i+1))),mask_length)
            for i in range(N)
        ])

        self.lin_down=nn.Linear(64,32)
        self.tanh=nn.Tanh()
        self.active_tanh=active_tanh

    def forward(self,spa,label):
        out=spa
        for layer in self.stages:
            out=layer(out,label)
        out=self.lin_down(out)
        if self.active_tanh:
            out=self.tanh(out)

        return out
        
"""
Teacher 网络 Decoder 部分
"""
# class Up_Decoder_Label(nn.Module):
#     def __init__(self,mask_length,dmodel,upsample_rate):
#         super(Up_Decoder_Label,self).__init__()
#         self.dim1=int(dmodel/upsample_rate)
#         self.dim2=int(self.dim1/upsample_rate)
#         self.lin_up=nn.Linear(32,self.dim2)
#         self.block2 = Block_label(self.dim2, self.dim1, mask_length)
#         self.block3 = Block_label(self.dim2, dmodel, mask_length)

#     def forward(self,spa,label):
        
#         out1=self.lin_up(spa)
#         out2=self.block2(out1,label)
#         out3=self.block3(out2,label)
        
#         return out3

class Up_Decoder_Label(nn.Module):
    def __init__(self,mask_length,dmodel,upsample_rate):
        super(Up_Decoder_Label,self).__init__()
        N=int(math.log(int(dmodel/64), int(upsample_rate)))

        self.stages=nn.ModuleList([
            Block_label(int(dmodel/int(math.pow(upsample_rate,N-i))),int(dmodel/int(math.pow(upsample_rate,N-i-1))),mask_length)
            for i in range(N)
        ])

        self.lin_up=nn.Linear(32,64)

    def forward(self,spa,label):
        out=self.lin_up(spa)
        for layer in self.stages:
            out=layer(out,label)
        
        return out


"""
Student 网络 Encoder 部分
"""
# class Down_Encoder_Spa(nn.Module):
#     def __init__(self, mask_length,dmodel,downsample_rate,active_tanh):
#         super(Down_Encoder_Spa,self).__init__()
#         self.dim1=int(dmodel/downsample_rate)
#         self.dim2=int(self.dim1/downsample_rate)
#         self.block1 = Block_spa(dmodel, self.dim1, mask_length)
#         self.block2 = Block_spa(self.dim1, self.dim2, mask_length)
#         self.lin_down=nn.Linear(self.dim2,32)
#         self.tanh=nn.Tanh()
#         self.active_tanh=active_tanh
        
#     def forward(self, spa):
#         out1 = self.block1(spa,spa)
#         out2 = self.block2(out1,spa)
#         out2=self.lin_down(out2)
#         if self.active_tanh:
#             out2=self.tanh(out2)
#         return out2

class Down_Encoder_Spa(nn.Module):
    def __init__(self, mask_length,dmodel,downsample_rate,Enc_active,n_layers):
        super(Down_Encoder_Spa,self).__init__()
        N=int(math.log(int(dmodel/64), int(downsample_rate)))

        self.stages=nn.ModuleList([
            Block_spa(int(dmodel/int(math.pow(downsample_rate,i))),int(dmodel/int(math.pow(downsample_rate,i+1))),mask_length,n_layers)
            for i in range(N)
        ])

        self.lin_down=nn.Linear(64,32)
        self.tanh=nn.Tanh()
        self.active_tanh=Enc_active
        
    def forward(self, spa):
        out=spa
        for layer in self.stages:
            out=layer(out)
        out=self.lin_down(out)
        if self.active_tanh:
            out=self.tanh(out)

        return out


"""
Student 网络 Decoder 部分
"""

# class Up_Decoder_Spa(nn.Module):
#     def __init__(self, mask_length,dmodel,upsample_rate):
#         super(Up_Decoder_Spa,self).__init__()
#         self.dim1=int(dmodel/upsample_rate)
#         self.dim2=int(self.dim1/upsample_rate)
#         self.block2 = Block_spa(self.dim2, self.dim1, mask_length)
#         self.block3 = Block_spa(self.dim1, dmodel, mask_length)
#         self.lin_up=nn.Linear(32,self.dim2)
#         # self.tanh=nn.Tanh()
#         # self.norm = nn.LayerNorm(32)
        
#     def forward(self, spa):
#         #out1 = self.block1(spa,spa)
#         out1=self.lin_up(spa)
#         out2 = self.block2(out1,spa)
#         out3 = self.block3(out2,spa)
#         return out

class Up_Decoder_Spa(nn.Module):
    def __init__(self, mask_length,dmodel,upsample_rate,n_layers):
        super(Up_Decoder_Spa,self).__init__()
        N=int(math.log(int(dmodel/64), int(upsample_rate)))

        self.stages=nn.ModuleList([
            Block_spa(int(dmodel/int(math.pow(upsample_rate,N-i))),int(dmodel/int(math.pow(upsample_rate,N-i-1))),mask_length,n_layers)
            for i in range(N)
        ])

        self.lin_up=nn.Linear(32,64)

    def forward(self,spa):
        out=self.lin_up(spa)
        for layer in self.stages:
            out=layer(out)
        
        return out




"""
Teacher 网络
"""
class MM_model(nn.Module):#teacher 
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active):
        super(MM_model,self).__init__()

        self.Encoder=Down_Encoder_Label(mask_length,dmodel,sample_rate,Enc_active)
        self.Decoder=Up_Decoder_Label(mask_length,dmodel,sample_rate)
        self.lin_cla=nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                   nn.ReLU(),
                                   nn.Linear(int(dmodel/2), 7))

        self.Spa_PE=FixedPositionalEncoding(embedding_dim=dmodel)

    def forward(self,spa,label):
        latent_feature=self.Encoder(spa,label)
        out1=self.Decoder(latent_feature,label)
        out_phase=self.lin_cla(out1)

        return latent_feature,out1,out_phase


"""
Stage2: student的f=32 feature 丢到teacher的decoder里面
"""
class MM_stage2_model_spa(nn.Module):
    def __init__(self, mask_length,dmodel,sample_rate,Enc_active):
        super(MM_stage2_model_spa, self).__init__()
        self.Encoder = Down_Encoder_Label(mask_length,dmodel,sample_rate,Enc_active)
        self.Decoder = Up_Decoder_Label(mask_length,dmodel,sample_rate)
        self.lin_cla=nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                   nn.ReLU(),
                                   nn.Linear(int(dmodel/2), 7))

        self.Spa_PE = FixedPositionalEncoding(embedding_dim=dmodel)


    def forward(self,latent1, label):
        out = self.Decoder(latent1,label)
        out_phase = self.lin_cla(out)

        return out, out_phase

#student的1024feature输入到teacher的mlp里面
class MM_stage2_3_model_spa_only_decoder(nn.Module):#student的1024feature输入到teacher的mlp里面
    def __init__(self, mask_length,dmodel,sample_rate,Enc_active):
        super(MM_stage2_3_model_spa_only_decoder, self).__init__()
        self.Encoder = Down_Encoder_Label(mask_length,dmodel,sample_rate,Enc_active)
        self.Decoder = Up_Decoder_Label(mask_length,dmodel,sample_rate)
        self.lin_cla=nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                   nn.ReLU(),
                                   nn.Linear(int(dmodel/2), 7))

        self.Spa_PE = FixedPositionalEncoding(embedding_dim=dmodel)

    def forward(self, latent1):
        out_phase1 = self.lin_cla(latent1)

        return out_phase1#1s是代表者只有spa


# only student model


class Student(nn.Module):
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active,n_layers):
        super(Student,self).__init__()

        self.Encoder=Down_Encoder_Spa(mask_length,dmodel,sample_rate,Enc_active,n_layers)
        self.Decoder=Up_Decoder_Spa(mask_length,dmodel,sample_rate,n_layers)
        self.lin_cla = nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(dmodel/2), 7))
   

    def forward(self,spa):
        latent_feature = self.Encoder(spa)
        out = self.Decoder(latent_feature)
    
        

        return latent_feature,out

class Student_KL(nn.Module):
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active,n_layers):
        super(Student_KL,self).__init__()

        self.Encoder=Down_Encoder_Spa(mask_length,dmodel,sample_rate,Enc_active,n_layers)
        self.Decoder=Up_Decoder_Spa(mask_length,dmodel,sample_rate,n_layers)
        self.lin_cla = nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(dmodel/2), 7))
   

    def forward(self,spa):
        latent_feature = self.Encoder(spa)
        out = self.Decoder(latent_feature)
        phase =self.lin_cla(out)
    
        

        return latent_feature,out,phase



class Student_new(nn.Module):
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active,n_layers):
        super(Student_new,self).__init__()

        self.Encoder=Down_Encoder_Spa(mask_length,dmodel,sample_rate,Enc_active,n_layers)
        self.Decoder=Up_Decoder_Spa(mask_length,dmodel,sample_rate,n_layers)
        self.lin_cla = nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(dmodel/2), 7))
   

    def forward(self,spa):
        latent_feature = self.Encoder(spa)
        out = self.Decoder(latent_feature)
        out_phase=self.lin_cla(out)
    
        return out_phase
    

class Student_LSC(nn.Module):
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active,n_layers):
        super(Student_LSC,self).__init__()

        self.Encoder=Down_Encoder_Spa(mask_length,dmodel,sample_rate,Enc_active,n_layers)
        self.Decoder=Up_Decoder_Spa(mask_length,dmodel,sample_rate,n_layers)
        self.lin_cla = nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(dmodel/2), 7))
   

    def forward(self,spa):
        latent_feature = self.Encoder(spa)
        out = self.Decoder(latent_feature)
        out_phase=self.lin_cla(out)
    
        return out_phase,latent_feature
    
class Student_tsne(nn.Module):
    def __init__(self,mask_length,dmodel,sample_rate,Enc_active,n_layers):
        super(Student_tsne,self).__init__()

        self.Encoder=Down_Encoder_Spa(mask_length,dmodel,sample_rate,Enc_active,n_layers)
        self.Decoder=Up_Decoder_Spa(mask_length,dmodel,sample_rate,n_layers)
        self.lin_cla = nn.Sequential(nn.Linear(dmodel, int(dmodel/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(dmodel/2), 7))
   

    def forward(self,spa):
        latent_feature = self.Encoder(spa)
        out = self.Decoder(latent_feature)
        out_phase=self.lin_cla(out)
    
        return latent_feature

#sup_con_loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):
        """
        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        batchsize是整个video的帧数
        """
        
        projections=projections.squeeze(0)

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
            torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )#exp

        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).cuda()
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).cuda()
        mask_combined = mask_similar_class * mask_anchor_out #不包括自己的所有的positive sample
        cardinality_per_samples = torch.sum(mask_combined, dim=1)#每个点的positive sample的数量

        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))#计算每（T，T）中每一个点的损失
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples##计算每一个时间点的loss
        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)#loss取平均

        return supervised_contrastive_loss

if __name__ == '__main__':

    band_width = 1000
    d_model = 1024
    down_rate = 2
    active_tanh = bool(1)

    enc_inputs = torch.ones((1,20,1024)).cuda()
    labels_phase=np.ones((20))
    labels_phase=torch.LongTensor(np.array(labels_phase))
    labels_phase=labels_phase.cuda()
    model = MM_model(mask_length=band_width,dmodel=d_model,sample_rate=down_rate,Enc_active=active_tanh)
    model=model.cuda()

    with torch.no_grad(): # 没有梯度优化，节约性能
        a,b,c = model(enc_inputs,labels_phase)
    
    # output1 shape:  torch.Size([1200, 32])
    # output2 shape:  torch.Size([1200, 1024])
    # output3 shape:  torch.Size([1200, 7])
    print("output1 shape: ", a.shape)
    print("output2 shape: ", b.shape)
    print("output3 shape: ", c.shape)

    parm={}
    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())
        parm[name]=parameters.detach().cpu().numpy()

    print("finish")




