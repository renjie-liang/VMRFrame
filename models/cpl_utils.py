            # 
# from models.loss import cal_nll_loss
import torch
from torch import nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None, fast_weights=None,
                gauss_weight=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)

            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2) == 1,
                    float('-1e30'),
                ).type_as(attn_weights)  # FP16 support: cast to float and back
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        from fairseq import utils
        attn_weights = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)

        if gauss_weight is not None:
            # gauss_weight = gauss_weight.unsqueeze(1).repeat(self.num_heads, tgt_len, 1)
            gauss_weight = gauss_weight.unsqueeze(1).unsqueeze(1)\
                .expand(-1, self.num_heads, tgt_len, -1).reshape(*attn_weights.shape)
            attn_weights = attn_weights * (gauss_weight + 1e-10)
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True)
        
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        d_model = d_model
        num_heads = num_heads
        self.dropout = dropout
        self.self_attn = MultiheadAttention(d_model, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_attn = MultiheadAttention(d_model, num_heads)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model << 1)
        self.fc2 = nn.Linear(d_model << 1, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, encoder_out=None, encoder_mask=None, self_attn_mask=None, 
                src_gauss_weight=None, tgt_gauss_weight=None):
        res = x
        x, weight = self.self_attn(x, x, x, mask, attn_mask=self_attn_mask, gauss_weight=tgt_gauss_weight)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            res = x
            x, weight = self.encoder_attn(x, encoder_out, encoder_out, encoder_mask, gauss_weight=src_gauss_weight)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = res + x
            x = self.encoder_attn_layer_norm(x)

        res = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = res + x
        x = self.final_layer_norm(x)
        return x, weight


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout=0.0, future_mask=True):
        super().__init__()
        self.future_mask = future_mask
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def buffered_future_mask(self, tensor):
        if not self.future_mask:
            return None
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(fill_with_neg_inf(tensor.new(dim, dim)), 1)
        if self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def forward(self, src, src_mask, tgt, tgt_mask, src_gauss_weight=None, tgt_gauss_weight=None):
        non_pad_src_mask = None if src_mask is None else 1 - src_mask
        non_pad_tgt_mask = None if tgt_mask is None else 1 - tgt_mask

        if src is not None:
            src = src.transpose(0, 1)

        x = tgt.transpose(0, 1)
        for layer in self.decoder_layers:
            x, weight = layer(x, non_pad_tgt_mask,
                              src, non_pad_src_mask,
                              self.buffered_future_mask(x), 
                              src_gauss_weight, tgt_gauss_weight)
        return x.transpose(0, 1), weight


# def infer_cpl(output, vmask, configs): ## don't consider vmask
#     P = configs.cpl.num_props
#     B = output['tlogist_prop'].size(0)  // P
    

#     tmask_props = torch.repeat_interleave(output['tmask'], P, dim=0)
#     word_ids = torch.repeat_interleave(output['word_ids'], P, dim=0)
#     nll_loss, acc = cal_nll_loss(output['tlogist_prop'], word_ids, tmask_props)
#     idx = nll_loss.view(B, P).argsort(dim=-1)

#     width = output['width'].view(B, P).gather(index=idx, dim=-1)
#     center = output['center'].view(B, P).gather(index=idx, dim=-1)
    
#     selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
#                                     torch.clamp(center+width/2, max=1)], dim=-1)
#     selected_props = selected_props.detach().cpu().numpy()

#     start_fracs, end_fracs = selected_props[:, 0, 0], selected_props[:, 0, 1] 

#     return start_fracs, end_fracs, selected_props
        

def infer_cpl(output, vmask, configs): ## don't consider vmask
    P = configs.cpl.num_props
    B = output['words_logit'].size(0)  // P
    

    tmask_props = torch.repeat_interleave(output['words_mask'], P, dim=0)
    word_ids = torch.repeat_interleave(output['word_ids'], P, dim=0)
    nll_loss, acc = cal_nll_loss(output['words_logit'], word_ids, tmask_props)
    idx = nll_loss.view(B, P).argsort(dim=-1)

    width = output['width'].view(B, P).gather(index=idx, dim=-1)
    center = output['center'].view(B, P).gather(index=idx, dim=-1)
    
    selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
                                    torch.clamp(center+width/2, max=1)], dim=-1)
    selected_props = selected_props.detach().cpu().numpy()

    start_fracs, end_fracs = selected_props[:, 0, 0], selected_props[:, 0, 1] 

    return start_fracs, end_fracs, selected_props
        






    # for bid, batch in enumerate(self.test_loader, 1):
    #     durations = np.asarray([i[1] for i in batch['raw']])
    #     gt = np.asarray([i[2] for i in batch['raw']])

    #     net_input = move_to_cuda(batch['net_input'])


    #     output = self.model(epoch=epoch, **net_input)
    #     B = len(durations)
    #     P = self.model.P
    #     k = min(P, 5)
        
    #     words_mask = output['words_mask'].unsqueeze(1).expand(B, P, -1).contiguous().view(B*P, -1)
    #     words_id = output['words_id'].unsqueeze(1).expand(B, P, -1).contiguous().view(B*P, -1)

    #     nll_loss, acc = cal_nll_loss(output['words_logit'], words_id, words_mask)
    #     idx = nll_loss.view(B, P).argsort(dim=-1)

    #     width = output['width'].view(B, P).gather(index=idx, dim=-1)
    #     center = output['center'].view(B, P).gather(index=idx, dim=-1)
    #     selected_props = torch.stack([torch.clamp(center-width/2, min=0), 
    #                                     torch.clamp(center+width/2, max=1)], dim=-1)
    #     selected_props = selected_props.cpu().numpy()
    #     gt = gt / durations[:, np.newaxis]
        
    #     res = top_1_metric(selected_props[:, 0], gt)
        
    #     for key, v in res.items():
    #         metrics_logger['R@1,'+key].update(v, B)
    #     res = top_n_metric(selected_props[:, :k].transpose(1, 0, 2), gt)
    #     for key, v in res.items():
    #         metrics_logger['R@%d,'%(k)+key].update(v, B)

    # msg = '|'.join([' {} {:.4f} '.format(k, v.avg) for k, v in metrics_logger.items()])
    # info('|'+msg+'|')
    # return metrics_logger



def plot_proposal(props, gt, img_name):
    from matplotlib import pyplot as plt

    plt.plot(gt, [1,1], color="black", linewidth="2")
    
    h = 1
    for i in range(len(props)):
        p = props[i]
        h -= 0.1
        plt.plot(p, [h, h], linestyle=":")
    plt.title(img_name)
    print("plot {}".format(img_name))
    plt.savefig("/storage/rjliang/3_ActiveLearn/seqpan_pytorch/imgs/CPL_proposal/{}.jpg".format(img_name))
    plt.cla()


def plot_proposal_batch(props_batch, records):
    for i in range(len(records)):
        sta_gtfrac = records[i]['s_time']/records[i]["duration"]
        end_gtfrac = records[i]['e_time']/records[i]["duration"]
        plot_proposal( props_batch[i], [sta_gtfrac, end_gtfrac], records[i]["vid"]+ "_" + str(i))
    

def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask



class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number



class DualTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_decoder_layers1, num_decoder_layers2, dropout=0.0):
        super().__init__()
        self.decoder1 = TransformerDecoder(num_decoder_layers1, d_model, num_heads, dropout)
        self.decoder2 = TransformerDecoder(num_decoder_layers2, d_model, num_heads, dropout)

    def forward(self, src1, src_mask1, src2, src_mask2, decoding, enc_out=None, gauss_weight=None, need_weight=False):
        assert decoding in [1, 2]
        if decoding == 1:
            if enc_out is None:
                enc_out, _ = self.decoder2(None, None, src2, src_mask2)
            out, weight = self.decoder1(enc_out, src_mask2, src1, src_mask1)
        elif decoding == 2:
            if enc_out is None:
                enc_out, _ = self.decoder1(None, None, src1, src_mask1, tgt_gauss_weight=gauss_weight)
                # enc_out = self.decoder1(None, None, src1, src_mask1)
            out, weight = self.decoder2(enc_out, src_mask1, src2, src_mask2, src_gauss_weight=gauss_weight)
        
        if need_weight:
            return enc_out, out, weight
        return enc_out, out


def cal_nll_loss(logit, idx, mask, weights=None):
    eps = 0.1
    acc = (logit.max(dim=-1)[1]==idx).float()
    mean_acc = (acc * mask).sum() / mask.sum()
    
    logit = logit.log_softmax(dim=-1)
    nll_loss = -logit.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
    smooth_loss = -logit.sum(dim=-1)
    nll_loss = (1 - eps) * nll_loss + eps / logit.size(-1) * smooth_loss
    if weights is None:
        nll_loss = nll_loss.masked_fill(mask == 0, 0)
        nll_loss = nll_loss.sum(dim=-1) / mask.sum(dim=-1)
    else:
        nll_loss = (nll_loss * weights).sum(dim=-1)

    return nll_loss.contiguous(), mean_acc


def rec_loss(words_logit, words_id, words_mask, num_props, configs, ref_words_logit=None):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    nll_loss = nll_loss.view(bsz, num_props)
    min_nll_loss = nll_loss.min(dim=-1)[0]

    final_loss = min_nll_loss.mean()

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask) 
        final_loss = final_loss + ref_nll_loss.mean()
        final_loss = final_loss / 2
    
    loss_dict = {
        'final_loss': final_loss.item(),
        'nll_loss': min_nll_loss.mean().item(),
    }
    if ref_words_logit is not None:
        loss_dict.update({
            'ref_nll_loss': ref_nll_loss.mean().item(),
            })

    return final_loss, loss_dict

    
def ivc_loss(words_logit, words_id, words_mask, num_props, configs, gauss_weight, neg_words_logit_1=None, neg_words_logit_2=None, ref_words_logit=None):
    bsz = words_logit.size(0) // num_props
    words_mask1 = words_mask.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)
    words_id1 = words_id.unsqueeze(1) \
        .expand(bsz, num_props, -1).contiguous().view(bsz*num_props, -1)

    nll_loss, acc = cal_nll_loss(words_logit, words_id1, words_mask1)
    min_nll_loss, idx = nll_loss.view(bsz, num_props).min(dim=-1)

    if ref_words_logit is not None:
        ref_nll_loss, ref_acc = cal_nll_loss(ref_words_logit, words_id, words_mask)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        ref_loss = torch.max(min_nll_loss - ref_nll_loss + configs["margin_1"], tmp_0)
        rank_loss = ref_loss.mean()
    else:
        rank_loss = min_nll_loss.mean()
    
    if neg_words_logit_1 is not None:
        neg_nll_loss_1, neg_acc_1 = cal_nll_loss(neg_words_logit_1, words_id1, words_mask1)
        neg_nll_loss_1 = torch.gather(neg_nll_loss_1.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_1 = torch.max(min_nll_loss - neg_nll_loss_1 + configs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_1.mean()
    
    if neg_words_logit_2 is not None:
        neg_nll_loss_2, neg_acc_2 = cal_nll_loss(neg_words_logit_2, words_id1, words_mask1)
        neg_nll_loss_2 = torch.gather(neg_nll_loss_2.view(bsz, num_props), index=idx.unsqueeze(-1), dim=-1).squeeze(-1)
        tmp_0 = torch.zeros_like(min_nll_loss).cuda()
        tmp_0.requires_grad = False
        neg_loss_2 = torch.max(min_nll_loss - neg_nll_loss_2 + configs["margin_2"], tmp_0)
        rank_loss = rank_loss + neg_loss_2.mean()

    loss = configs['alpha_1'] * rank_loss

    gauss_weight = gauss_weight.view(bsz, num_props, -1)
    gauss_weight = gauss_weight / gauss_weight.sum(dim=-1, keepdim=True)
    target = torch.eye(num_props).unsqueeze(0).cuda() * configs["lambda"]
    source = torch.matmul(gauss_weight, gauss_weight.transpose(1, 2))
    div_loss = torch.norm(target - source, dim=(1, 2))**2

    loss = loss + configs['alpha_2'] * div_loss.mean()

    return loss, {
        'ivc_loss': loss.item(),
        'neg_loss_1': neg_loss_1.mean().item() if neg_words_logit_1 is not None else 0.0,
        'neg_loss_2': neg_loss_2.mean().item() if neg_words_logit_2 is not None else 0.0,
        'ref_loss': ref_loss.mean().item() if ref_words_logit is not None else 0.0,
        'div_loss': div_loss.mean().item()
    }