import pdb
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
import numpy as np
import argparse
from datetime import datetime

from scipy.interpolate import interp1d
from scipy.optimize import linear_sum_assignment

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class AuxilliaryEncoderCMT(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderCMT, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []
        # pdb.set_trace()

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class AuxilliaryEncoderST(nn.TransformerEncoder):
    def __init__(self, encoder_layer_local, num_layers, norm=None):
        super(AuxilliaryEncoderST, self).__init__(encoder_layer=encoder_layer_local,
                                            num_layers=num_layers,
                                            norm=norm)

    def forward(self, src, mask=None, src_key_padding_mask=None, get_attn=False):
        output = src
        attn_matrices = []

        for i, mod in enumerate(self.layers):
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class LearnedIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.seq_len = seq_len
        self.person_encoding = nn.Embedding(1000, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:
        x = x + self.person_encoding(torch.arange(num_people).repeat_interleave(self.seq_len, dim=0).to(self.device)).unsqueeze(1)
        return self.dropout(x)


class LearnedTrajandIDEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=21, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.seq_len = seq_len
        self.learned_encoding = nn.Embedding(self.seq_len, d_model//2, max_norm=True).to(device)
        self.person_encoding = nn.Embedding(1000, d_model//2, max_norm=True).to(device)

    def forward(self, x: torch.Tensor, num_people=1) -> torch.Tensor:

        half = x.size(3)//2 ## 124

        x[:,:,:,0:half*2:2] = x[:,:,:,0:half*2:2] + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)
        x[:,:,:,1:half*2:2] = x[:,:,:,1:half*2:2] + self.person_encoding(torch.arange(num_people).unsqueeze(0).repeat_interleave(self.seq_len, dim=0).to(self.device)).unsqueeze(0)


        return self.dropout(x)

class Learnedbb3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.seq_len = seq_len
        self.learned_encoding = nn.Embedding(self.seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class Learnedbb2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=9, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.seq_len = seq_len
        self.learned_encoding = nn.Embedding(self.seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.learned_encoding(torch.arange(self.seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class Learnedpose3dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.seq_len = seq_len
        self.learned_encoding = nn.Embedding(self.seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class Learnedpose2dEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, seq_len=198, device='cuda:0'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.device = device
        self.learned_encoding = nn.Embedding(seq_len, d_model, max_norm=True).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)
        x = x + self.learned_encoding(torch.arange(seq_len).to(self.device)).unsqueeze(1).unsqueeze(0)

        return self.dropout(x)

class TransMotion(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0'):

        super(TransMotion, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        # self.joints_pose = 22
        self.obs_and_pred = obs_and_pred
        self.device = device

        self.fc_in_traj = nn.Linear(2,nhid)
        # self.fc_out_traj = nn.Linear(nhid, 2)
        self.fc_out_traj_sc = nn.Linear(nhid, 4)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)
        self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)

        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)

        # self.fc_in_3dbb = nn.Linear(4,nhid)
        # self.bb3d_encoder = Learnedbb3dEncoding(nhid, dropout, device=device)

        # self.fc_in_2dbb = nn.Linear(4,nhid)
        # self.bb2d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        self.fc_in_1ds = nn.Linear(1,nhid)
        self.s1d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        self.fc_in_1dc = nn.Linear(1,nhid)
        self.c1d_encoder = Learnedbb2dEncoding(nhid, dropout, device=device)

        # self.fc_in_3dpose = nn.Linear(3, nhid)
        # self.pose3d_encoder = Learnedpose3dEncoding(nhid, dropout, device=device)

        # self.fc_in_2dpose = nn.Linear(2, nhid)
        # self.pose2d_encoder = Learnedpose2dEncoding(nhid, dropout, device=device)


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)



    def forward(self, tgt, padding_mask,metamask=None):
        # import pdb
        # pdb.set_trace()
        B, in_F, NJ, K = tgt.shape  # torch.Size([16, 9, 38, 4])

        F = self.obs_and_pred
        J = self.token_num

        out_F = F - in_F
        N = NJ // J

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]
        tgt = tgt.reshape(B,F,N,J,K)

        ## add mask
        mask_ratio_traj = 0.0
        mask_ratio_modality = 0.0

        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        # # tgt_2dbb = tgt[:,:,:,1,:4].to(self.device)
        # tgt_1ds = tgt[:, :, :, 1, :1].to(self.device)
        # tgt_1dc = tgt[:, :, :, 2, :1].to(self.device)


        ## mask for specific modality for whole observation horizon
        # modality_selection_2dbb = (torch.rand((B,1,N)).float().to(self.device) > mask_ratio_modality).unsqueeze(3).repeat(1,F,1,4)
        # tgt_vis = tgt_2dbb*modality_selection_2dbb
        # tgt_2dbb = tgt_vis.to(self.device)
        modality_selection_1ds = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection_1dc = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection = torch.cat((modality_selection_1ds, modality_selection_1dc),3)
        tgt_vis = tgt[:,:,:,1:]*modality_selection
        tgt_1ds = tgt_vis[:,:,:,0,:1].to(self.device)
        tgt_1dc = tgt_vis[:,:,:,1,:1].to(self.device)



        ############
        # Transformer
        ###########
        # pdb.set_trace()
        tgt_traj = self.fc_in_traj(tgt_traj)
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)

        # tgt_2dbb = self.fc_in_2dbb(tgt_2dbb[:,:in_F])
        # tgt_2dbb = self.bb2d_encoder(tgt_2dbb)
        # import pdb
        # pdb.set_trace()
        tgt_1ds = self.fc_in_1ds(tgt_1ds[:,:in_F])
        tgt_1ds = self.s1d_encoder(tgt_1ds)

        tgt_1dc = self.fc_in_1dc(tgt_1dc[:,:in_F])
        tgt_1dc = self.c1d_encoder(tgt_1dc)

        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)

        tgt_1ds = torch.transpose(tgt_1ds,0,1).reshape(in_F,-1,self.nhid)
        tgt_1dc = torch.transpose(tgt_1dc,0,1).reshape(in_F,-1,self.nhid)
        tgt = torch.cat((tgt_traj, tgt_1ds, tgt_1dc),0)  # torch.Size([39, 80, 128])
        # tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
        # tgt = torch.cat((tgt_traj,tgt_2dbb),0)

        # tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)
        # tgt_3dbb = torch.transpose(tgt_3dbb,0,1).reshape(in_F,-1,self.nhid)
        # tgt_2dbb = torch.transpose(tgt_2dbb,0,1).reshape(in_F,-1,self.nhid)
        # tgt_3dpose = torch.transpose(tgt_3dpose, 0,1).reshape(in_F*self.joints_pose, -1, self.nhid)
        # tgt_2dpose = torch.transpose(tgt_2dpose, 0,1).reshape(in_F*self.joints_pose, -1, self.nhid)

        # tgt = torch.cat((tgt_traj,tgt_3dbb,tgt_2dbb,tgt_3dpose,tgt_2dpose),0)
        # import pdb
        # pdb.set_trace()
        out_local = self.local_former(tgt, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        # pdb.set_trace()

        ##### local residual ######
        out_local = out_local * self.output_scale + tgt  # torch.Size([48, 304, 128])
        # pdb.set_trace()
#
        out_local = out_local[:F].reshape(F,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)  # torch.Size([741, 16, 128])  B=16
        # pdb.set_trace()
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)

        ##### global residual ######
        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]
        # out_primary = self.fc_out_traj(out_primary)
        # out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)
        out_primary = self.fc_out_traj_sc(out_primary)
        out = out_primary.transpose(0, 1).reshape(B, F, 1, 4)

        return out


class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, seq_len, emb_dim]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        b, seq_len, hidden_dim = x.shape

        # x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        # x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(b, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(b, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.emb_dim)   # torch.Size([16, 21, 128])

        # print(out.shape)

        return out, att_weights



class MSTABlock(nn.Module):
    def __init__(self, seq_len=21, nhid=128):
        super(MSTABlock, self).__init__()
        self.project_in = nn.Linear(seq_len * nhid, nhid)
        self.bn1d = nn.BatchNorm1d(nhid)
        self.relu = nn.LeakyReLU()
        self.cross_att_global = Cross_MultiAttention(in_channels=seq_len, emb_dim=nhid, num_heads=8, att_dropout=0.0, aropout=0.0)
        self.project_low = nn.Linear(2 * nhid, nhid)
        self.project_out = nn.Linear(nhid, 1)

    def forward(self, x):
        x_in = self.project_in(x.reshape(x.shape[0], -1))  # 1,16,128
        x_in = self.relu(self.bn1d(x_in).squeeze(0))
        ######## 分类代码
        # # 两两拼接
        # i, j = torch.meshgrid(torch.arange(x_in.shape[0]), torch.arange(x_in.shape[0]))
        # x_cat = torch.cat((x_in[i.flatten()], x_in[j.flatten()]), dim=1)
        # # 投射到低维空间
        # x_low = self.relu(self.bn1d(self.project_low(x_cat)))
        # # 进行分类
        # x_out = self.project_out(x_low)  # (16*16)*2
        # return x_out
        ######## 表示学习
        # pdb.set_trace()
        # out, att_weights = self.cross_att_global(x_in, x_in)  # 1,8,16,16
        # return torch.sum(att_weights, dim=1)
        return x_in



class AssoMOT(nn.Module):
    def __init__(self, tok_dim=21, nhid=256, nhead=4, dim_feedfwd=1024, nlayers_local=2, nlayers_global=4, dropout=0.1, activation='relu', output_scale=1, obs_and_pred=21,  num_tokens=47, device='cuda:0'):

        super(AssoMOT, self).__init__()
        self.seq_len = tok_dim
        self.nhid = nhid
        self.output_scale = output_scale
        self.token_num = num_tokens
        # self.joints_pose = 22
        self.obs_and_pred = obs_and_pred
        self.device = device

        self.fc_in_traj = nn.Linear(2,nhid)
        self.fc_out_traj = nn.Linear(nhid, 2)
        self.double_id_encoder = LearnedTrajandIDEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)
        self.id_encoder = LearnedIDEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)

        self.scale = torch.sqrt(torch.FloatTensor([nhid])).to(device)

        self.fc_in_1ds = nn.Linear(1,nhid)
        self.s1d_encoder = Learnedbb2dEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)

        self.fc_in_1dc = nn.Linear(1,nhid)
        self.c1d_encoder = Learnedbb2dEncoding(nhid, dropout, seq_len=self.obs_and_pred, device=device)


        encoder_layer_local = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.local_former = AuxilliaryEncoderCMT(encoder_layer_local, num_layers=nlayers_local)

        encoder_layer_global = nn.TransformerEncoderLayer(d_model=nhid,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedfwd,
                                                   dropout=dropout,
                                                   activation=activation)
        self.global_former = AuxilliaryEncoderST(encoder_layer_global, num_layers=nlayers_global)
        self.mata_block = MSTABlock(seq_len=self.obs_and_pred, nhid=self.nhid)

    def forward(self, tgt, padding_mask,metamask=None):
        # pdb.set_trace()
        B, in_F, NJ, K = tgt.shape  # torch.Size([16, 9, 38, 4])

        F = self.obs_and_pred
        J = self.token_num

        out_F = F - in_F
        N = NJ // J

        ## keep padding
        pad_idx = np.repeat([in_F - 1], out_F)
        i_idx = np.append(np.arange(0, in_F), pad_idx)
        tgt = tgt[:,i_idx]
        # pdb.set_trace()
        tgt = tgt.reshape(B,F,N,J,K)

        ## add mask
        mask_ratio_traj = 0.0
        mask_ratio_modality = 0.0

        tgt_traj = tgt[:,:,:,0,:2].to(self.device)
        traj_mask = torch.rand((B,F,N)).float().to(self.device) > mask_ratio_traj
        traj_mask = traj_mask.unsqueeze(3).repeat_interleave(2,dim=-1)
        tgt_traj = tgt_traj*traj_mask

        ## mask for specific modality for whole observation horizon
        modality_selection_1ds = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection_1dc = (torch.rand((B,1,N,1)).float().to(self.device) > mask_ratio_modality).unsqueeze(4).repeat(1,F,1,1,4)
        modality_selection = torch.cat((modality_selection_1ds, modality_selection_1dc),3)
        tgt_vis = tgt[:,:,:,1:]*modality_selection
        tgt_1ds = tgt_vis[:,:,:,0,:1].to(self.device)
        tgt_1dc = tgt_vis[:,:,:,1,:1].to(self.device)

        ############
        # Transformer
        ###########
        # pdb.set_trace()
        tgt_traj = self.fc_in_traj(tgt_traj)  # torch.Size([16, 21, 4, 2])
        # pdb.set_trace()
        tgt_traj = self.double_id_encoder(tgt_traj, num_people=N)  # torch.Size([16, 21, 4, 128])
        # pdb.set_trace()

        # pdb.set_trace()
        tgt_1ds = self.fc_in_1ds(tgt_1ds)
        tgt_1ds = self.s1d_encoder(tgt_1ds)  # torch.Size([16, 21, 4, 128])

        tgt_1dc = self.fc_in_1dc(tgt_1dc)
        tgt_1dc = self.c1d_encoder(tgt_1dc)  # torch.Size([16, 21, 4, 128])

        tgt_padding_mask_global = padding_mask.repeat_interleave(F, dim=1)
        tgt_padding_mask_local = padding_mask.reshape(-1).unsqueeze(1).repeat_interleave(self.seq_len,dim=1)
        tgt_feature = tgt_traj[:, :, 0, :] + tgt_1ds[:, :, 0, :] + tgt_1dc[:, :, 0, :]  # torch.Size([16, 21, 4, 128])
        out_asso = self.mata_block(x=tgt_feature)
        # tgt_feature = torch.cat((tgt_traj[:, :, 0, :], tgt_1ds[:, :, 0, :], tgt_1dc[:, :, 0, :]),0)  # torch.Size([48, 21, 4, 128])

        tgt_traj = torch.transpose(tgt_traj,0,1).reshape(F,-1,self.nhid)  # torch.Size([21, 64, 128])
        tgt_1ds = torch.transpose(tgt_1ds,0,1).reshape(F,-1,self.nhid)  # torch.Size([21, 64, 128])
        tgt_1dc = torch.transpose(tgt_1dc,0,1).reshape(F,-1,self.nhid)  # torch.Size([21, 64, 128])

        # pdb.set_trace()

        # import pdb
        # pdb.set_trace()
        tgt_traj = torch.cat((tgt_traj, tgt_1ds, tgt_1dc),0)  # torch.Size([63, 64, 128])
        # pdb.set_trace()
        out_local = self.local_former(tgt_traj, mask=None, src_key_padding_mask=tgt_padding_mask_local)
        # pdb.set_trace()
        ##### local residual ######
        out_local = out_local * self.output_scale + tgt_traj  # torch.Size([63, 64, 128])
        # pdb.set_trace()
        out_local = out_local[:F].reshape(F,B,N,self.nhid).permute(2,0,1,3).reshape(-1,B,self.nhid)  # torch.Size([84, 16, 128])  B=16
        # pdb.set_trace()
        out_global = self.global_former(out_local, mask=None, src_key_padding_mask=tgt_padding_mask_global)  # torch.Size([84, 16, 128])  B=16
        # 加一个Muliti-source Tracklets Association Block (MSTABlock)  <tgt_feature, out_global>
        tgt_feature_enhanced = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]  # torch.Size([21, 16, 128])  B=16
        tgt_feature_enhanced = torch.transpose(tgt_feature_enhanced,0,1)  # torch.Size([16, 21, 128])  B=16
        # pdb.set_trace()
        # out_asso = self.mata_block(x=tgt_feature_enhanced + tgt_feature)
        # pdb.set_trace()

        # tgt_cat_feature = torch.cat((tgt_feature, tgt_feature_enhanced), 0)  # torch.Size([64, 21, 128])  B=16 64=16*4

        ##### global residual ######
        out_global = out_global * self.output_scale + out_local
        out_primary = out_global.reshape(N,F,out_global.size(1),self.nhid)[0]  # torch.Size([4, 21, 16, 128])  B=16
        # tgt_feature_enhanced.shape = F,B,self.nhid

        out_primary = self.fc_out_traj(out_primary)
        out = out_primary.transpose(0, 1).reshape(B, F, 1, 2)  # F = history + prediction

        return out, out_asso


def create_model(config):
    seq_len = config["MODEL"]["seq_len"]
    token_num = config["MODEL"]["token_num"]
    nhid=config["MODEL"]["dim_hidden"]
    nhead=config["MODEL"]["num_heads"]
    nlayers_local=config["MODEL"]["num_layers_local"]
    nlayers_global=config["MODEL"]["num_layers_global"]
    dim_feedforward=config["MODEL"]["dim_feedforward"]

    if config["MODEL"]["type"] == "transmotion":
        # logger.info("Creating bert model.")
        model = AssoMOT(tok_dim=seq_len,
            nhid=nhid,
            nhead=nhead,
            dim_feedfwd=dim_feedforward,
            nlayers_local=nlayers_local,
            nlayers_global=nlayers_global,
            output_scale=config["MODEL"]["output_scale"],
            obs_and_pred=config["TRAIN"]["input_track_size"] + config["TRAIN"]["output_track_size"],
            num_tokens=token_num,
            device=config["DEVICE"]
        ).to(config["DEVICE"]).float()
    else:
        raise ValueError(f"Model type '{config['MODEL']['type']}' not found")

    return model

def interp_points(data_array_tmp, mode=0):
    points = data_array_tmp[:, 1:]
    # 拆分time, x, y, sog, cog
    times = points[:, 0]
    x_coords = points[:, 1]
    y_coords = points[:, 2]
    sog = points[:, 3]
    cog = points[:, 4]
    # 创建插值函数
    x_interp_func = interp1d(times, x_coords, kind='linear', fill_value="extrapolate")
    y_interp_func = interp1d(times, y_coords, kind='linear', fill_value="extrapolate")
    sog_interp_func = interp1d(times, sog, kind='linear', fill_value="extrapolate")
    cog_interp_func = interp1d(times, cog, kind='linear', fill_value="extrapolate")
    # import pdb
    # pdb.set_trace()

    missing_time_range = np.setdiff1d(np.arange(times[0], times[-1] + 1, 1), times)
    if mode == 0:
        # 用全0来填充
        missing_MMSI = np.array(list(data_array_tmp[0, 0] for i in range(missing_time_range.shape[0])))
        # 对缺失时间点进行插值
        missing_x = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_y = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_sog = np.array(list(0 for i in range(missing_time_range.shape[0])))
        missing_cog = np.array(list(0 for i in range(missing_time_range.shape[0])))
    elif mode == 1:
        missing_MMSI = np.array(list(data_array_tmp[0, 0] for i in range(missing_time_range.shape[0])))
        # 对缺失时间点进行插值
        missing_x = x_interp_func(missing_time_range)
        missing_y = y_interp_func(missing_time_range)
        missing_sog = sog_interp_func(missing_time_range)
        missing_cog = cog_interp_func(missing_time_range)
    # 将插值结果合并到缺失点
    missing_points = np.column_stack((missing_MMSI, missing_time_range, missing_x, missing_y, missing_sog, missing_cog))
    # import pdb
    # pdb.set_trace()

    # 将插值后的点与原始点结合
    all_points = np.vstack((data_array_tmp, missing_points))

    # 按时间排序
    all_points = all_points[np.argsort(all_points[:, 1])]
    return all_points


def inference(model, config, input_joints, padding_mask, out_len=14):
    model.eval()

    with torch.no_grad():

        pred_joints, asso_feature = model(input_joints, padding_mask)

    pred_output_joints = pred_joints[:,-out_len:]
    refine_output_joints = pred_joints[:, :-out_len]
    # pdb.set_trace()

    return pred_output_joints, refine_output_joints, asso_feature

def hungarian_matching_traj(array1, array2):
    """
    使用匈牙利匹配算法对两个点集合进行匹配。

    参数:
    obs_array_t1 (numpy.ndarray): 观测点集合，形状为 (n, 2)。
    pred_array_t1 (numpy.ndarray): 预测点集合，形状为 (m, 2)。

    返回:
    list: 匹配对的索引列表，每个元素为 (obs_idx, pred_idx)。
    """
    # 计算距离矩阵
    array1 = array1.astype(np.float32)
    cost_matrix = np.linalg.norm(array1[:, np.newaxis] - array2, axis=2)
    # # 通过距离最小原则找到匹配结果， array2 的点可能被多个 array1 匹配上
    # matched_pairs = []
    # for i in range(cost_matrix.shape[0]):
    #     j = np.argmin(cost_matrix[i])
    #     matched_pairs.append([i, j])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # import pdb
    # pdb.set_trace()
    # 构建匹配结果
    matched_pairs = list(zip(row_ind, col_ind))
    return matched_pairs, cost_matrix


# def hungarian_matching_traj(obs_array_t1, pred_array_t1):
#     """
#     使用匈牙利匹配算法对两个点集合进行匹配。

#     参数:
#     obs_array_t1 (numpy.ndarray): 观测点集合，形状为 (n, 2)。
#     pred_array_t1 (numpy.ndarray): 预测点集合，形状为 (m, 2)。

#     返回:
#     list: 匹配对的索引列表，每个元素为 (obs_idx, pred_idx)。
#     """
#     # 计算距离矩阵
#     obs_array_t1 = obs_array_t1.astype(np.float32)
#     cost_matrix = np.linalg.norm(obs_array_t1[:, np.newaxis] - pred_array_t1, axis=2)

#     # 使用匈牙利算法找到最小成本匹配
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     # import pdb
#     # pdb.set_trace()
#     # 构建匹配结果
#     matched_pairs = list(zip(row_ind, col_ind))
#     return matched_pairs, cost_matrix


def hungarian_matching_mmsi(gt_mmsi, pred_mmsi):
    """
    使用匈牙利匹配算法对两个点集合进行匹配。

    参数:
    obs_array_t1 (numpy.ndarray): 观测点集合，形状为 (n, 2)。
    pred_array_t1 (numpy.ndarray): 预测点集合，形状为 (m, 2)。

    返回:
    list: 匹配对的索引列表，每个元素为 (obs_idx, pred_idx)。
    """
    # 计算距离矩阵
    cost_matrix = np.linalg.norm(gt_mmsi[:, np.newaxis, np.newaxis] - pred_mmsi[:, np.newaxis], axis=2)

    # 使用匈牙利算法找到最小成本匹配
    # import pdb
    # pdb.set_trace()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # import pdb
    # pdb.set_trace()
    # 构建匹配结果
    matched_pairs = list(zip(row_ind, col_ind))
    return matched_pairs, cost_matrix


def hungarian_matching(outputs):
    """
    使用匈牙利匹配算法对两个点集合进行匹配。

    参数:
    obs_array_t1 (numpy.ndarray): 观测点集合，形状为 (n, 2)。
    pred_array_t1 (numpy.ndarray): 预测点集合，形状为 (m, 2)。

    返回:
    list: 匹配对的索引列表，每个元素为 (obs_idx, pred_idx)。
    """
    # 计算距离矩阵
    # pdb.set_trace()
    cost_matrix = np.linalg.norm(outputs[:, np.newaxis, np.newaxis] - outputs[:, np.newaxis], axis=3)[:, :, 0]
    diagonal_matrix = np.diag(torch.ones(outputs.shape[0]) * 1e9)
    cost_matrix += diagonal_matrix
    # 使用匈牙利算法找到最小成本匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # 构建匹配结果
    matched_pairs = list(zip(row_ind, col_ind))
    return matched_pairs, cost_matrix










