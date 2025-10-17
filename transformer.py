import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=512):
        super().__init__()
        # shape: [mx_seq_len,1]
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        item = 1/10000**(torch.arange(0, d_model, 2)/d_model)
        tmp_pos = position*item
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(tmp_pos)
        pe[:, 1::2] = torch.cos(tmp_pos)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, False)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        pe = self.pe
        return x+pe[:, :seq_len, :]


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)  # 将得分转换为概率分布，在最后一个维度

    def forward(self, Q, K, V, mask=None):
        # X:batch, seq_len, d_model
        # d_model:embedding向量的维度
        # Q，query向量 维度：batch，heads，seq_len_q,d_k
        # k，key向量 维度：batch，heads，seq_len_k,d_k
        # v，value向量 维度：batch，heads，seq_len_v,d_v
        # mask:那些位置要看，哪些位置要忽略
        d_k = Q.size(-1)  # q的最后一维是对每一个query向量的维度，代表对每个query进行缩放
        # batch，heads，seq_len_q,d_k,   batch，heads,d_k,seq_len_k ---> batch，heads,seq_len_q,seq_len_k
        score = torch.matmul(Q, K.transpose(-2, -1))/math.sqrt(d_k)  # 进行缩放，让梯度更稳定
        # 如果提供了mask，则通过mask==0来找到需要屏蔽的位置，mask_fill会将这些为宗旨的值改为-inf(负无穷)
        # 然后经过softmax之后值会变为0
        # 设置mask==0 表示被屏蔽 mask==1表示当前位置可见
        if mask is not None:
            # print("mask shape:", mask.shape)
            score = score.masked_fill(mask == 0, float('-inf'))
        # batch，heads,seq_len_q,seq_len_k 对最后一维进行softmax，即对key进行，得到注意力权重矩阵，对每一个query的key权重之和为1
        att = self.softmax(score)
        att = self.dropout(att)  # 对注意力权重进行dropout，防止过拟合
        # att:batch，heads,seq_len_q,seq_len_k; V:batch，heads，seq_len_v,d_v ---> att*V: batch，heads，seq_len_q, d_v
        out = torch.matmul(att, V)
        return out, att


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        # d_model: embedding的维度 512
        # n_head: 多头注意力的头数 8
        # d_model需要被 n_head整除 结果为64
        assert d_model % n_head == 0
        self.d_k = d_model // n_head  # 每个头的维度
        self.n_head = n_head

        # 将输入映射到Q， K， V三个向量，通过线性映射让模型具有学习能力
        self.W_q = nn.Linear(d_model, d_model)  # query的线性映射，维度不需要改变，方便后续的多头拆分
        self.W_k = nn.Linear(d_model, d_model)  # key的线性映射
        self.W_v = nn.Linear(d_model, d_model)  # value的线性映射
        self.fc = nn.Linear(d_model, d_model)   # 多头拼接后再映射回原来的d_model，让模型融合不同头的信息

        self.attention = SelfAttention(dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)   # 用于残差后的归一化

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)  # 获取batch的大小
        # q 的维度 batch，seq_len,d_model -->batch,seq_len,self.n_head,self.d_k --> batch,self.n_head,seq_len,self.d_k
        # 为了让每个注意力头独立处理整个序列，方便后续计算注意力权重
        Q = self.W_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 计算注意力
        out, att = self.attention(Q, K, V, mask)    # att为注意力权重， out为注意力加权后的值
        # out.transpose(1, 2): batch，heads，seq_len_q, d_v --> batch，seq_len_q, heads，d_v --> batch,seq_len,d_model
        # contiguous目的是让tensor在内存中连续存储，避免view的时候产生报错
        # 多头拼接
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.d_k)
        out = self.fc(out)  # 让输入输出一致，方便残差连接
        out = self.dropout(out)
        # 残差连接+layernorm
        # return self.norm(out+q), att  # 返回输出和注意力权重
        return self.norm(out+q)  # 返回输出和注意力权重


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # d_ff: 前馈隐藏层维度
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_multi_head_att = MultiHeadAttention(d_model=d_model, n_head=heads, dropout=dropout)
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for i in range(2)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        multi_head_att = self.self_multi_head_att(x, x, x, mask)
        # multi_head_att = self.norm[0](x + multi_head_att)
        multi_head_att = multi_head_att

        ffn_out = self.ffn(multi_head_att)
        ffn_out = self.norm[1](multi_head_att + ffn_out)
        out = self.dropout(ffn_out)
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, num_layer, heads, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encode = PositionalEmbedding(d_model, max_seq_len)

        self.encode_layer = nn.ModuleList([EncoderLayer(heads, d_model, d_ff, dropout) for i in range(num_layer)])

    def forward(self, x, src_mask):
        embed_x = self.embedding(x)
        pos_encode_x = self.position_encode(embed_x)
        for layer in self.encode_layer:
            pos_encode_x = layer(pos_encode_x, src_mask)
        return pos_encode_x


class DecoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.masked_att = MultiHeadAttention(d_model=d_model, n_head=heads, dropout=dropout)
        self.att = MultiHeadAttention(d_model=d_model, n_head=heads, dropout=dropout)
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for i in range(3)])
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encode_kv, dst_mask=None, src_dst_mask=None):
        masked_att_out = self.masked_att(x, x, x, dst_mask)
        # masked_att_out = self.norms[0](x + masked_att_out)
        masked_att_out = masked_att_out

        att_out = self.att(masked_att_out, encode_kv, encode_kv, src_dst_mask)
        att_out = self.norms[1](masked_att_out + att_out)
        ffn_out = self.ffn(att_out)
        ffn_out = self.norms[2](att_out + ffn_out)
        out = self.dropout(ffn_out)
        return out


class Decoder(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model, num_layer, heads, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, pad_idx)
        self.position_encode = PositionalEmbedding(d_model, max_seq_len)

        self.decode_layer = nn.ModuleList([DecoderLayer(heads, d_model, d_ff, dropout) for i in range(num_layer)])

    def forward(self, x, encoder_kv, dst_mask=None, src_dst_mask=None):
        embed_x = self.embedding(x)
        pos_encode_x = self.position_encode(embed_x)
        for layer in self.decode_layer:
            pos_encode_x = layer(pos_encode_x, encoder_kv, dst_mask, src_dst_mask)
        return pos_encode_x


class Transformer(nn.Module):
    def __init__(self, encode_vocab_size, decode_vocab_size, pad_idx, d_model, num_layer, heads, d_ff, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.encoder = Encoder(encode_vocab_size, pad_idx, d_model, num_layer, heads, d_ff, dropout, max_seq_len)
        self.decoder = Decoder(decode_vocab_size, pad_idx, d_model, num_layer, heads, d_ff, dropout, max_seq_len)
        self.linear = nn.Linear(d_model, decode_vocab_size)
        self.pad_idx = pad_idx

    def generate_mask(self, q, k, is_triu_mask=False):
        """
        :param q: batch, seq_len
        :param k: batch, seq_len
        :return:
        """
        device = q.device
        batch, seq_q = q.shape
        _, seq_k = k.shape
        # batch, head, seq_q,seq_k
        mask = (k != self.pad_idx).unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch, 1, seq_q, seq_k).to(device)
        if is_triu_mask:
            dst_triu_mask = torch.triu(torch.ones(seq_q, seq_k, dtype=torch.bool), diagonal=1)
            dst_triu_mask = dst_triu_mask.unsqueeze(0).unsqueeze(1).expand(batch, 1, seq_q, seq_k).to(device)
            return mask | dst_triu_mask
        return mask

    def forward(self, src, dst):
        src_mask = self.generate_mask(src, src)
        encoder_out = self.encoder(src, src_mask)
        dst_mask = self.generate_mask(dst,dst,True)
        src_dst_mask = self.generate_mask(dst,src)
        decoder_out = self.decoder(dst, encoder_out, dst_mask, src_dst_mask)
        out = self.linear(decoder_out)
        return out


if __name__ == "__main__":
    att = Transformer(100, 200, 0, 512, 6, 8, 1024, 0.1)
    x = torch.randint(0, 100, (4, 64))
    y = torch.randint(0, 200, (4, 64))
    out = att(x, y)
    print(out.shape)









