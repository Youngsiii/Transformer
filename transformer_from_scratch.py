import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        """
        注意力模块
        embed_size: 每个token的嵌入维度
        num_heads: 多头注意力的头数
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads

        assert embed_size % num_heads == 0, "Embedding size need to be divisible by num_heads"
        self.head_dim = embed_size // num_heads   # 每个头的嵌入维度

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        # values: (BS, val_len, embed_size)
        # keys: (BS, key_len, embed_size)
        # queries: (BS, query_len, embed_size)
        BS = queries.shape[0]

        val_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = self.values(values)     # (BS, val_len, embed_size)
        keys = self.keys(keys)           # (BS, key_len, embed_size)
        queries = self.queries(queries)  # (BS, query_len, embed_size)

        values = values.reshape(BS, val_len, self.num_heads, self.head_dim)       # (BS, val_len, num_heads, head_dim)
        keys = keys.reshape(BS, key_len, self.num_heads, self.head_dim)           # (BS, key_len, num_heads, head_dim)
        queries = queries.reshape(BS, query_len, self.num_heads, self.head_dim)   # (BS, query_len, num_heads, head_dim)

        values = values.permute(0, 2, 1, 3)    # (BS, num_heads, val_len, head_dim)
        keys = keys.permute(0, 2, 1, 3)        # (BS, num_heads, key_len, head_dim)
        queries = queries.permute(0, 2, 1, 3)  # (BS, num_heads, query_len, head_dim)

        energy = queries @ keys.transpose(-2, -1)
        # Q(K^T)  (BS, num_heads, query_len, head_dim) @ (BS, num_heads, head_dim, key_len) -> (BS, num_heads, query_len, key_len)

        if mask is not None:     # 需要添加mask，mask的形状与energy一样为(query_len, key_len)
            energy = energy.masked_fill(mask == 0, float("-1e20"))
            # energy: (BS, num_heads, query_len, key_len)
            # src_mask: (BS, 1, 1, src_len) 会自动广播到(BS, num_heads, src_len, src_len)  其实这个mask没有完全将填充部分的注意力给掩码掉，只掩码了一部分
            # trg_mask: (BS, 1, trg_len, trg_len) 会自动广播到(BS, num_heads, trg_len, trg_len)
            # trg_mask是一个下对角矩阵(trg_len, trg_len)，主对角线上方全为0，可以使得当前token只能获得当前token之前的注意力信息而无法获得当前token之后的token信息
            # src_mask主要是针对不同长度的输入，希望屏蔽掉pad的token信息

        attention = torch.softmax(energy / ((self.head_dim) ** 0.5), dim=-1)  # (BS, num_heads, query_len, key_len)
        out = attention @ values    # key_len = val_len
        # (BS, num_heads, query_len, key_len) @ (BS, num_heads, val_len, head_dim) -> (BS, num_heads, query_len, head_dim)

        out = out.permute(0, 2, 1, 3)  # (BS, query_len, num_heads, head_dim)
        out = out.reshape(BS, query_len, -1)   # (BS, query_len, embed_size)
        # self-attention模块输出与queries的形状相同为(query_len, embed_size)

        out = self.fc_out(out)   # (BS, query_len, embed_size)
        return out


# model = SelfAttention(embed_size=128, num_heads=8)
# values = keys = queries = torch.randn(32, 15, 128)
# print(model(values, keys, queries, None).shape)  # 输出应该与queries形状相同(BS, query_len, embed_size)=(32, 15, 128)





class EncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.attention = SelfAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)   # 这两个LayerNorm的参数是不一样的，需要设置两个LayerNorm
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size * forward_expansion, embed_size),
        )

    def forward(self, values, keys, queries, mask):
        # values: (BS, val_len, embed_size)
        # keys: (BS, key_len, embed_size)
        # queries: (BS, query_len, embed_size)
        attention = self.attention(values, keys, queries, mask)   # attention的形状与queries一样为(BS, query_len, embed_size)
        x = self.dropout(self.norm1(attention + queries))         # (BS, query_len, embed_size)   +queries操作可用于Decoder中交互注意力模块
        out = self.feed_forward(x)      # (BS, query_len, embed_size)->(BS, query_len, embed_size*forward_expansion)->(BS, query_len, embed_size)
        out = self.dropout(self.norm2(out + x))     # (BS, query_len, embed_size)
        return out   # (BS, query_len, embed_size)   Encoder输出与queries的形状相同为(BS, query_len, embed_size)





class Encoder(nn.Module):
    def __init__(self, src_vocab_size, max_length, embed_size, num_heads, num_layers, dropout, forward_expansion, device):
        """
        构造完整的编码器结构
        src_vocab_size: 源语言的词汇表大小，用于词嵌入  nn.Embedding(src_vocab_size, embed_size)
        max_length: 输入句子的最大长度/最大单词数，用于位置嵌入/位置编码  nn.Embedding(max_length, embed_size)
        embed_size: 词嵌入的维度
        num_heads: 多头注意力的头数
        num_layers: 编码器的个数
        dropout: 用于dropout的比例
        forward_expansion: 编码器中MLP中的隐藏层相对于embed_size扩大的倍数
        device: 编写下面代码过程中会出现新构造的张量，需要将其转移到device上才能与模块中的其他张量进行计算
        """
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device  # 保存device，用于后续新构造的张量转移到device上
        self.dropout = nn.Dropout(dropout)


        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)   # 词嵌入，词汇表大小为src_vocab_size，每个词嵌入的维度为embed_size，构成一个词嵌入矩阵，后续根据每个词的索引来提取对应的词嵌入向量
        self.position_embedding = nn.Embedding(max_length, embed_size)   # 位置嵌入，句子长度为max_length，每个位置嵌入的维度为embed_size，构成一个位置嵌入矩阵，后续根据位置索引来提取位置嵌入向量

        self.layers = nn.ModuleList([EncoderBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_layers)])


    def forward(self, x, mask):
        # 输入x: (BS, src_len)  BS为batch_size即每个batch中有多少个句子, seq_length表示每个句子有多少单词（这里的单词其实是每个单词对应的索引）
        # 第1句 ['word1', 'word2', ..., 'wordseq_lenght']    # 其实是wordi对应的张量torch.tensor(wordi)
        # 第2句 ['word1', 'word2', ..., 'wordseq_lenght']    # 其实是wordi对应的张量torch.tensor(wordi)
        # ......
        # 第BS句 ['word1', 'word2', ..., 'wordseq_lenght']    # 其实是wordi对应的张量torch.tensor(wordi)
        BS, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(BS, seq_length).to(self.device)   # (BS, src_len)  构造的新张量需要转移到device上才能与模块的其他张量进行计算
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))  # out: (BS, src_len, embed_size)
        # x: (BS, src_len)->(BS, src_len, embed_size)
        # positions: (BS, src_len)->(BS, src_len, embed_size)

        for layer in self.layers:
            out = layer(out, out, out, mask)    # (BS, src_len, embed_size)

        return out    # (BS, src_len, embed_size)



class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = SelfAttention(embed_size, num_heads)
        self.encoder_block = EncoderBlock(embed_size, num_heads, dropout, forward_expansion)

    def forward(self, x, values, keys, src_mask, trg_mask):
        # x: DecoderBlock的输入 (BS, trg_len, embed_size)
        # values, keys: Encoder的输出 (BS, src_len, embed_size)
        # trg_mask: target语言的mask，用于Decoder第一个Attention的mask
        # src_mask: 用于Decoder的交互Attention的mask
        queries = self.dropout(self.norm(self.attention(x, x, x, trg_mask) + x))   # (BS, trg_len, embed_size)
        out = self.encoder_block(values, keys, queries, src_mask)     # (BS, trg_len, embed_size)
        return out     # (BS, trg_len, embed_size)


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, max_length, embed_size, num_heads, num_layers, dropout, forward_expansion, device):
        """
        构造解码器模块
        trg_vocab_size: 目标语言词汇表大小，用于目标语言的词嵌入 nn.Embedding(trg_vocab_size, embed_size)
        max_length: 目标语言句子的最大长度/最大单词数，用于位置嵌入
        embed_size: 词嵌入的维度，也是位置嵌入的维度，便于位置嵌入与词嵌入相加
        num_heads: 多头注意力的头数
        num_layers: 解码器的个数
        dropout: dropout的比例
        forward_expansion: MLP中隐藏层相对于embed_size扩大的倍数
        device: 下面的代码会新构造张量，需要将新构造的张量转移到device上才能与模块中的其他张量进行计算
        """
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, num_heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)   # 将embed_size维度变成trg_vocab_size维度，代表每个类别的概率
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        # x: Decoder输入 (BS, trg_len)
        # enc_out: Encoder输出 (BS, src_len, embed_size)
        # trg_mask: target语言的mask，用于Decoder的第一个Attention的mask
        # src_mask: 用于Decoder的交互Attention的mask
        BS, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(BS, seq_length).to(self.device)    # (BS, trg_len)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))   # out: (BS, trg_len, embed_size)
        # x: (BS, trg_len)->(BS, trg_len, embed_size)
        # positions: (BS, trg_len)->(BS, trg_len, embed_size)
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, src_mask, trg_mask)   # (BS, trg_len, embed_size)

        out = self.fc_out(out)    # (BS, trg_len, embed_size)->(BS, trg_len, trg_vocab_size)
        return out    # (BS, trg_len, trg_vocab_size)


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            max_length=100,
            embed_size=512,
            num_heads=8,
            num_layers=6,
            forward_expansion=4,
            dropout=0,
            device="cpu",
    ):
        """
        src_vocab_size: 源语言词汇表大小
        trg_vocab_size: 目标语言词汇表大小
        src_pad_idx: 源语言句子src的填充词汇索引一般为0
        trg_pad_idx: 目标语言句子trg的填充词汇索引一般也为0
        max_length: 句子的最大长度，用于位置编码
        embed_size: 词嵌入的维度
        num_heads: 多头注意力的头数
        num_layers: 编码器和解码器的个数
        forward_expansion: MLP隐藏层相对与embed_size的倍数
        dropout: dropout的比例
        device: "cpu" or "cuda"
        """
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx  # 例如0
        self.trg_pad_idx = trg_pad_idx  # 例如0
        self.encoder = Encoder(src_vocab_size, max_length, embed_size, num_heads, num_layers, dropout, forward_expansion, device)
        self.decoder = Decoder(trg_vocab_size, max_length, embed_size, num_heads, num_layers, dropout, forward_expansion, device)



    # def make_trg_mask(self, trg):  # trg: (BS, trg_len)
    #     BS, trg_len = trg.shape
    #     trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(BS, 1, trg_len, trg_len)
    #     return trg_mask

    def make_src_mask(self, src):   # src:(BS, src_len)
        # 例如： src=[[88, 75, 33, 25, 0],[3, 83, 25, 0, 0],[11, 22, 0, 0, 0],[22, 22, 22, 22, 22],[77, 23, 18, 12, 0]]
        # (src != 0) -> [[T, T, T, T, F],[T, T, T, F, F],[T, T, F, F, F],[T, T, T, T, T],[T, T, T, T, F]]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)     # (BS, 1, 1, src_len)    加到注意力(相似度)上时会自动广播为(BS, num_heads, src_len, src_len)
        return src_mask   # (BS, 1, 1, src_len)  # 其实这个mask没有完全将填充部分的注意力给掩码掉，只掩码了一部分


    def make_trg_mask(self, trg):   # trg:(BS, trg_len)
        BS, trg_len = trg.shape
        # torch.tril产生下三角矩阵，即矩阵主对角线上方全为0
        trg_mask = torch.tril(torch.ones(trg_len, trg_len)).expand(BS, 1, trg_len, trg_len)   # (trg_len, trg_len)->(BS, 1, trg_len, trg_len)
        return trg_mask   # (BS, 1, trg_len, trg_len)  # 这个mask可以防止当前的token获得之后token的信息




    def forward(self, src, trg):     # src:(BS, src_len)  trg:(BS, trg_len)
        src_mask = self.make_src_mask(src)      # (BS, 1, 1, src_len)
        trg_mask = self.make_trg_mask(trg)      # (BS, 1, trg_len, trg_len)
        enc_out = self.encoder(src, src_mask)   # (BS, src_len, embed_size)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)   # (BS, trg_len, trg_vocab_size)
        return out     # (BS, trg_len, trg_vocab_size)



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Transformer的源输入src形状为(BS, src_len)，其中每个数表示源语言每个单词的索引，目标输入trg形状也为(BS, trg_len)，每个数表示目标语言每个单词的索引
    src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0],
                        [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)    # (BS, src_len)=(2, 9)
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0],
                        [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)       # (BS, trg_len)=(2, 8)
    src_vocab_size = 10  # 源语言词汇表大小，假设只有10个单词
    trg_vocab_size = 10  # 目标语言词汇表大小，也假设只有10个单词

    src_pad_idx = 0  # 源语言句子src填充词汇索引设为0，src每个句子中最后的0都表示填充词汇
    trg_pad_idx = 0  # 目标语言句子trg填充词汇索引也设为0，trg每个句子中最后的0都表示填充词汇

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(device)
    out = model(src, trg)
    print(out.shape)  # 应该是(BS, trg_len, trg_vocab_size)=(2, 8, 10)












