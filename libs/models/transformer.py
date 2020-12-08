"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

class InteractionLayer(nn.Module):
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.d_feature = d_feature

        # self.det_tfm = nn.Linear(d_model, d_feature)
        # self.rel_tfm = nn.Linear(d_model, d_feature)
        # self.det_value_tfm = nn.Linear(d_model, d_feature)

        # self.rel_norm = nn.LayerNorm(d_model)

        # if dropout is not None:
        #     self.dropout = dropout
        #     self.det_dropout = nn.Dropout(dropout)
        #     self.rel_add_dropout = nn.Dropout(dropout)
        # else:
        #     self.dropout = None

    def forward(self, det_in, puppet_in):
        if puppet_in.sum() == 0:
            return det_in, det_in

        puppet_out = (det_in + puppet_in) // 2
        return det_in, puppet_out


class PuppetTransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 puppet_decoder_layer,
                 num_layers,
                 interaction_layer=None,
                 norm=None,
                 puppet_norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.puppet_layers = _get_clones(puppet_decoder_layer, num_layers)
        self.num_layers = num_layers
        if interaction_layer is not None:
            self.puppet_interaction_layers = _get_clones(interaction_layer, num_layers)
        else:
            self.puppet_interaction_layers = None
        self.norm = norm
        self.puppet_norm = puppet_norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, puppet_tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                puppet_tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                puppet_memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                puppet_tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                puppet_query_pos: Optional[Tensor] = None):
        output = tgt
        puppet_output = puppet_tgt

        bs = memory.shape[1] // 2
        det_memory = memory[..., :bs, :]
        puppet_memory = memory[..., bs:, :]
        det_memory_key_padding_mask = memory_key_padding_mask[:bs, :]
        puppet_memory_key_padding_mask = memory_key_padding_mask[bs:, :]
        det_pos = pos[..., :bs, :]
        puppet_pos = pos[..., bs:, :]

        intermediate = []
        puppet_intermediate = []

        for i in range(self.num_layers):
            output = self.layers[i](output, det_memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=det_memory_key_padding_mask,
                pos=det_pos, query_pos=query_pos)

            if self.puppet_interaction_layers is not None:
                output, puppet_output = self.puppet_interaction_layers[i](
                    output, puppet_output
                )
            puppet_output = self.puppet_layers[i](puppet_output, puppet_memory,
                tgt_mask=puppet_tgt_mask,
                memory_mask=puppet_memory_mask,
                tgt_key_padding_mask=puppet_tgt_key_padding_mask,
                memory_key_padding_mask=puppet_memory_key_padding_mask,
                pos=puppet_pos, query_pos=puppet_query_pos)

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                puppet_intermediate.append(self.puppet_norm(puppet_output))

        if self.norm is not None:
            output = self.norm(output)
            puppet_output = self.puppet_norm(puppet_output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                puppet_intermediate.pop()
                puppet_intermediate.append(puppet_output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(puppet_intermediate)

        return output, puppet_output


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_sub_decoder_layers=6, num_obj_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        num_decoder_layers = num_sub_decoder_layers
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape

        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        # memory shape: (W*H, bs, 256)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(cfg):
    # return Transformer(
    #     d_model=cfg.TRANSFORMER.HIDDEN_DIM,
    #     dropout=cfg.TRANSFORMER.DROPOUT,
    #     nhead=cfg.TRANSFORMER.NHEADS,
    #     dim_feedforward=cfg.TRANSFORMER.DIM_FEEDFORWARD,
    #     num_encoder_layers=cfg.TRANSFORMER.ENC_LAYERS,
    #     num_sub_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
    #     num_obj_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
    #     normalize_before=cfg.TRANSFORMER.PRE_NORM,
    #     return_intermediate_dec=True,
    # )
    return PuppetTransformer(
        d_model=cfg.TRANSFORMER.HIDDEN_DIM,
        dropout=cfg.TRANSFORMER.DROPOUT,
        nhead=cfg.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.TRANSFORMER.ENC_LAYERS,
        num_decoder_layers=cfg.TRANSFORMER.DEC_LAYERS,
        normalize_before=cfg.TRANSFORMER.PRE_NORM,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


if __name__ == "__main__":
    transformer = Transformer()
    import torch
    from torch.autograd import Variable

    from libs.utils.misc import NestedTensor
    from libs.models.position_encoding import PositionEmbeddingSine

    x = Variable(torch.randn(2, 256, 25, 38))
    mask = Variable(torch.ones(2, 25, 38).type(torch.BoolTensor))
    input_sample = NestedTensor(x, mask)
    query_embed = nn.Embedding(64, 256)
    pos_embed = PositionEmbeddingSine(num_pos_feats=128)
    pos = pos_embed(input_sample)

    sub_hs, obj_hs, memory = transformer(x, mask, query_embed.weight, pos)
    num, c, bs = sub_hs.shape
    query_embed_action = query_embed.weight.unsqueeze(-1).repeat(1, 1, bs)
    action_emb = torch.cat([sub_hs, obj_hs, query_embed_action], dim=1)
    print(action_emb.shape)
