import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import random
from models.deformable_transformer import MLP
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional
import fvcore.nn.weight_init as weight_init


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



class Noiser:
    def __init__(self, noise_ratio = 0.1):
        self.noise_ratio = noise_ratio
    
    def _wa_noise_forward(self, cur_embeds):
        # embeds (q, b, c), classes (q)
        cur_embeds = cur_embeds.permute(1,0,2)
        indices = list(range(cur_embeds.shape[0]))
        np.random.shuffle(indices)
        noise_init = cur_embeds[indices]
        weight_ratio = torch.rand(cur_embeds.shape[0], 1, 1)
        noise_init = cur_embeds * weight_ratio.to(cur_embeds) + noise_init * (1.0 - weight_ratio.to(cur_embeds))
        ret_indices = torch.arange(cur_embeds.shape[0], dtype=torch.int64).numpy()
        ret_indices[(weight_ratio[:, 0, 0] < 0.5).to(torch.bool).numpy()] =\
            np.array(indices)[(weight_ratio[:, 0, 0] < 0.5).to(torch.bool).numpy()]
        
        noise_init = noise_init.permute(1,0,2)
        return list(ret_indices), noise_init
    
    def match_embds_batch(self,cur_embds, ref_embds,cur_embeds_no_norm):
        cur_embds = cur_embds / cur_embds.norm(dim=-1)[:,:,None]
        ref_embds = ref_embds / ref_embds.norm(dim=-1)[:,:,None]
        
        # id_check = 0
        # print('check : ',cur_embds[0,0].sum())
        cos_sim = torch.einsum('bqc,bpc->bpq',cur_embds,ref_embds)
        C = 1 - cos_sim
        C = C.cpu()
        C = torch.where(torch.isnan(C), torch.full_like(C, 0), C)
        
        indices = [linear_sum_assignment(C[i].detach().numpy())[1] for i in range(C.shape[0])]
        out = torch.stack([cur_embeds_no_norm[i][indices[i]] for i in range(cur_embeds_no_norm.shape[0])])
        
        # print(cur_embds.sum())
        # print(indices,out.shape)
        # new_id_check = indices[0].tolist().index(id_check)
        # print('id : ',id_check,new_id_check)
        # print('check : ',out[0,new_id_check].sum())
        return indices,out
        
    
    def __call__(self, ref_embeds, cur_embeds, cur_embeds_no_norm=None, activate=False):
        if cur_embeds_no_norm is None:
            cur_embeds_no_norm = cur_embeds
        matched_indices, rerange_embds = self.match_embds_batch(ref_embeds, cur_embeds,cur_embeds_no_norm)
        if activate and random.random() < self.noise_ratio:
            indices, noise_init = self._wa_noise_forward(cur_embeds_no_norm)
            return indices, noise_init
        else:
            return matched_indices, rerange_embds

class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, key, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, key, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class ReferringTracker_noiser(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=1024,
        num_head=8,
        decoder_layer_num=6,
        noise_ratio=0.1,
    ):
        super(ReferringTracker_noiser, self).__init__()
        
        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
        
        self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)
        for layer in self.ref_proj.layers:
            weight_init.c2_xavier_fill(layer)

        self.noiser = Noiser(noise_ratio=noise_ratio)
    
    def forward(self,reference_embds, zoomin_embds):
        reference = self.ref_proj(reference_embds[-1])
        list_output = []
        
        for i in range(self.num_layers):
            if i == 0:
                # indices, noise_init = self.noiser(zoomin_embds[i],reference_embds[-1],activate=self.training) # phase 3

                indices, noise_init = self.noiser(reference_embds[-1],zoomin_embds[i],activate=self.training) # phase 4

                # output = self.transformer_cross_attention_layers[i](noise_init.transpose(0,1),reference.transpose(0,1),
                #                                                     zoomin_embds[i].transpose(0,1),zoomin_embds[i].transpose(0,1)).transpose(0,1)
                # noise_init = zoomin_embds[i] # phase 2
                output = self.transformer_cross_attention_layers[i](noise_init.transpose(0,1),zoomin_embds[i].transpose(0,1),
                                                                    reference_embds[-1].transpose(0,1),reference_embds[-1].transpose(0,1)).transpose(0,1)

                
                
                output = self.transformer_self_attention_layers[i](
                    output.transpose(0,1), tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                ).transpose(0,1)
                
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )
                list_output.append(output)
            else:
                # output = self.transformer_cross_attention_layers[i](list_output[-1].transpose(0,1),reference.transpose(0,1),
                #                                                     zoomin_embds[i].transpose(0,1),zoomin_embds[i].transpose(0,1)).transpose(0,1)

                output = self.transformer_cross_attention_layers[i](list_output[-1].transpose(0,1),zoomin_embds[i].transpose(0,1),
                                                                    reference_embds[-1].transpose(0,1),reference_embds[-1].transpose(0,1)).transpose(0,1)

                
                output = self.transformer_self_attention_layers[i](
                    output.transpose(0,1), tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=None
                ).transpose(0,1)
                
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )
                list_output.append(output)

        return torch.stack(list_output,dim=0) 

hs_zoomin = torch.randn(6,3,60,256)
hs_ref = torch.randn(6,3,60,256)
lvl = 0

noise = Noiser()
# indices = noise.match_embds(hs_zoomin[lvl],hs_ref[lvl])
# indices, noise_init = noise.match_embds_batch(hs_zoomin[lvl],hs_ref[lvl])


attn = SelfAttentionLayer(d_model=256,nhead=8,dropout=0.1)
cur_hs = hs_zoomin[lvl]
output = attn(cur_hs.transpose(0,1)).transpose(0,1)
print(cur_hs.shape,output.shape)

tracker = ReferringTracker_noiser()
hs = tracker(hs_ref,hs_zoomin)

print(hs.shape)

