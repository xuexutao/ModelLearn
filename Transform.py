import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model) # Query 变换
        self.W_k = nn.Linear(d_model, d_model) # Key 变换
        self.W_v = nn.Linear(d_model, d_model) # Value 变换
        self.W_o = nn.Linear(d_model, d_model) # 输出变换

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):  # batch_size, num_heads, seq_len_q, d_k
        # 计算注意力分数: Q * K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码 (如果提供)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) # 用一个很小的数填充被mask的位置

        # 计算注意力权重 (Softmax)
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights) # 应用 dropout

        # 计算注意力输出: weights * V
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, query, key, value, mask=None): # batch_size, seq_len_q, d_model
        batch_size = query.size(0)

        # 1. 线性变换并切分成多头
        # Q, K, V 形状: (batch_size, seq_len, d_model)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # 改变形状为 (batch_size, seq_len, num_heads, d_k) 并 transpose 成 (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 2. 计算缩放点积注意力
        # context 形状: (batch_size, num_heads, seq_len_q, d_k)
        # attn_weights 形状: (batch_size, num_heads, seq_len_q, seq_len_k)
        context, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出
        # context 形状变回 (batch_size, seq_len_q, num_heads, d_k)
        context = context.transpose(1, 2).contiguous()
        # context 形状变回 (batch_size, seq_len_q, d_model)
        context = context.view(batch_size, -1, self.d_model)

        # 4. 最终线性变换
        output = self.W_o(context)
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x -> linear1 -> relu -> dropout -> linear2
        output = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        # div_term 计算 log空间中的 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)

        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe 形状为 (max_len, d_model), 我们需要在 batch_size 维度上广播
        # 所以增加一个维度变为 (1, max_len, d_model) 以便与 (batch_size, seq_len, d_model) 相加
        pe = pe.unsqueeze(0) # (1, max_len, d_model)

        # 将 pe 注册为 buffer, 这样它不会被视为模型参数, 但会随模型移动 (e.g., to device)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 的 seq_len 不能超过 max_len
        # self.pe[:, :x.size(1), :] 选择与输入序列长度匹配的位置编码
        # 形状为 (1, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # 1. 多头自注意力子层
        # 残差连接: src + dropout(self_attn(norm1(src)))
        # 注意: Transformer 原始论文是 post-LN (LayerNorm 在残差之后), 但 pre-LN (LayerNorm 在残差之前) 实践中更稳定
        # 这里我们使用 pre-LN 的变体 (但代码结构更像 post-LN, 注意力输入是原始 src)
        # 或者严格按照论文的 post-LN:
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout1(attn_output) # 残差连接
        src = self.norm1(src) # 层归一化

        # 2. 位置前馈网络子层
        # 残差连接: src + dropout(feed_forward(norm2(src)))
        ff_output = self.feed_forward(src)
        src = src + self.dropout2(ff_output) # 残差连接
        src = self.norm2(src) # 层归一化

        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout) # 掩码多头自注意力
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout) # 编解码器注意力
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. 掩码多头自注意力子层 (Masked Multi-Head Self-Attention)
        # Q, K, V 都是 tgt
        attn_output, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = tgt + self.dropout1(attn_output) # 残差连接
        tgt = self.norm1(tgt) # 层归一化

        # 2. 多头编解码器注意力子层 (Multi-Head Encoder-Decoder Attention)
        # Query 是 tgt (来自上一个子层), Key 和 Value 是 memory (编码器输出)
        cross_attn_output, cross_attn_weights = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = tgt + self.dropout2(cross_attn_output) # 残差连接
        tgt = self.norm2(tgt) # 层归一化

        # 3. 位置前馈网络子层
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout3(ff_output) # 残差连接
        tgt = self.norm3(tgt) # 层归一化

        return tgt # 通常也会返回 cross_attn_weights 用于可视化或分析

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout) # Embedding 后的 dropout

    def forward(self, src, src_mask=None):
        # 1. 词嵌入和位置编码
        # src_embed 形状: (batch_size, src_seq_len, d_model)
        src_embed = self.embedding(src) * math.sqrt(self.d_model) # 乘以 sqrt(d_model) 是常见做法
        src_embed = self.pos_encoder(src_embed)
        src_embed = self.dropout(src_embed) # Dropout after embedding and positional encoding

        # 2. 通过 N 个编码器层
        output = src_embed
        for layer in self.layers:
            output = layer(output, src_mask)

        return output

class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, output_vocab_size, max_seq_len, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(output_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout) # Embedding 后的 dropout

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # 1. 词嵌入和位置编码
        # tgt_embed 形状: (batch_size, tgt_seq_len, d_model)
        tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_embed = self.pos_encoder(tgt_embed)
        tgt_embed = self.dropout(tgt_embed)

        # 2. 通过 N 个解码器层
        output = tgt_embed
        for layer in self.layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        return output

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, num_heads,
                 input_vocab_size, output_vocab_size, d_ff,
                 max_seq_len, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, input_vocab_size, max_seq_len, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, num_heads, d_ff, output_vocab_size, max_seq_len, dropout)
        self.fc_out = nn.Linear(d_model, output_vocab_size) # 最终输出层, 映射到词汇表

        # 初始化参数 (可选, 但通常有益)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src_tokens, pad_idx=0):
        src_mask = (src_tokens != pad_idx).unsqueeze(1).unsqueeze(2)
        # (batch_size, 1, 1, src_seq_len)
        return src_mask

    def make_tgt_mask(self, tgt_tokens, pad_idx=0):
        batch_size, tgt_len = tgt_tokens.shape
        # 1. Padding Mask (屏蔽 padding token)
        tgt_pad_mask = (tgt_tokens != pad_idx).unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, tgt_len)
                                                                      # 在 MultiHeadAttention 中会广播到 (batch_size, 1, tgt_len, tgt_len)

        # 2. Subsequent Mask (屏蔽未来 token, 用于自回归)
        # 创建一个上三角矩阵, 对角线以上为 1 (True), 以下为 0 (False)
        # 我们需要的是下三角矩阵 (对角线及以下为 True), 所以取反
        subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt_tokens.device)).bool()
        # (tgt_len, tgt_len)
        # 在 MultiHeadAttention 中会广播到 (batch_size, num_heads, tgt_len, tgt_len)

        # 合并两个掩码
        # tgt_pad_mask 形状 (batch_size, 1, 1, tgt_len)
        # subsequent_mask 形状 (tgt_len, tgt_len)
        # 我们希望最终的 tgt_mask 形状是 (batch_size, 1, tgt_len, tgt_len)
        # 注意: MultiHeadAttention 中的 mask 是 mask == 0 的地方被填充 -1e9
        # 所以 True 表示保留, False 表示屏蔽
        tgt_mask = tgt_pad_mask & subsequent_mask # & 是逐元素 AND
        return tgt_mask


    def forward(self, src_tokens, tgt_tokens, src_pad_idx=0, tgt_pad_idx=0):
        src_mask = self.make_src_mask(src_tokens, src_pad_idx)
        tgt_mask = self.make_tgt_mask(tgt_tokens, tgt_pad_idx)

        # memory_mask 应该与 src_mask 类似, 用于解码器关注编码器输出时屏蔽 padding
        # 在 MultiHeadAttention 的 scaled_dot_product_attention 中, mask 的形状是 (batch_size, num_heads, seq_len_q, seq_len_k)
        # 对于 encoder-decoder attention, Q 来自 decoder, K/V 来自 encoder.
        # seq_len_q 是 tgt_len, seq_len_k 是 src_len.
        # 所以 memory_mask 应该作用于 src_len 维度。
        # self.make_src_mask 返回的是 (batch_size, 1, 1, src_seq_len), 这是正确的。
        memory_mask = src_mask

        # 1. 编码器处理源序列
        # memory 形状: (batch_size, src_seq_len, d_model)
        memory = self.encoder(src_tokens, src_mask)

        # 2. 解码器处理目标序列和编码器输出
        # dec_output 形状: (batch_size, tgt_seq_len, d_model)
        dec_output = self.decoder(tgt_tokens, memory, tgt_mask, memory_mask)

        # 3. 最终线性层输出 logits
        # output_logits 形状: (batch_size, tgt_seq_len, output_vocab_size)
        output_logits = self.fc_out(dec_output)

        return output_logits

if __name__ == '__main__':
    # --- 模型参数 ---
    SRC_VOCAB_SIZE = 5000  # 源语言词汇表大小
    TGT_VOCAB_SIZE = 5000  # 目标语言词汇表大小
    D_MODEL = 512          # 模型维度
    NUM_HEADS = 8          # 多头注意力的头数
    NUM_ENCODER_LAYERS = 6 # 编码器层数
    NUM_DECODER_LAYERS = 6 # 解码器层数
    D_FF = 2048            # 前馈网络中间层维度
    MAX_SEQ_LEN = 100      # 序列最大长度
    DROPOUT = 0.1          # Dropout 比例
    PAD_IDX = 0            # Padding token 的 ID (假设为0)

    # --- 实例化模型 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transformer_model = Transformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        input_vocab_size=SRC_VOCAB_SIZE,
        output_vocab_size=TGT_VOCAB_SIZE,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    # --- 打印模型结构和参数量 ---
    print(transformer_model)
    total_params = sum(p.numel() for p in transformer_model.parameters())
    trainable_params = sum(p.numel() for p in transformer_model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")


    # --- 准备伪数据进行测试 ---
    BATCH_SIZE = 2
    SRC_SEQ_LEN = 10 # 源序列长度
    TGT_SEQ_LEN = 12 # 目标序列长度 (用于 teacher forcing)

    # 随机生成源序列和目标序列的 token ID (0 到 vocab_size-1 之间)
    # 确保 token ID 不超过词汇表大小
    src_tokens = torch.randint(1, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN), device=device) # 假设 0 是 padding
    tgt_tokens = torch.randint(1, TGT_VOCAB_SIZE, (BATCH_SIZE, TGT_SEQ_LEN), device=device) # 假设 0 是 padding

    # 模拟一些 padding
    src_tokens[0, -3:] = PAD_IDX
    tgt_tokens[1, -2:] = PAD_IDX

    print("\n--- 输入数据 ---")
    print(f"源序列 (src_tokens) 形状: {src_tokens.shape}")
    print(src_tokens)
    print(f"目标序列 (tgt_tokens) 形状: {tgt_tokens.shape}")
    print(tgt_tokens)

    # --- 创建掩码 (模型内部会自动创建, 这里只是为了演示) ---
    src_mask_example = transformer_model.make_src_mask(src_tokens, pad_idx=PAD_IDX)
    tgt_mask_example = transformer_model.make_tgt_mask(tgt_tokens, pad_idx=PAD_IDX)
    print("\n--- 掩码示例 (部分) ---")
    print(f"源掩码 (src_mask) 形状: {src_mask_example.shape}")
    # print(f"源掩码 (src_mask) [0,0,0,:]: {src_mask_example[0,0,0,:]}") # 打印第一个样本的padding mask
    print(f"目标掩码 (tgt_mask) 形状: {tgt_mask_example.shape}")
    # print(f"目标掩码 (tgt_mask) [0,0,:,:]: \n{tgt_mask_example[0,0,:,:]}") # 打印第一个样本的联合mask


    # --- 模型前向传播 ---
    print("\n--- 模型输出 ---")
    try:
        output_logits = transformer_model(src_tokens, tgt_tokens, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)
        print(f"输出 Logits 形状: {output_logits.shape}") # 应为 (BATCH_SIZE, TGT_SEQ_LEN, TGT_VOCAB_SIZE)

        # 简单检查输出值是否合理 (例如, 不是 NaN 或 Inf)
        if torch.isnan(output_logits).any() or torch.isinf(output_logits).any():
            print("警告: 输出包含 NaN 或 Inf!")
        else:
            print("输出值检查通过 (无 NaN/Inf)。")

    except Exception as e:
        print(f"模型前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()

    # --- 简单测试单个模块 (例如 MultiHeadAttention) ---
    print("\n--- 测试 MultiHeadAttention ---")
    multi_head_attn = MultiHeadAttention(d_model=D_MODEL, num_heads=NUM_HEADS).to(device)
    # 伪造 Q, K, V 输入 (batch_size, seq_len, d_model)
    q_test = torch.rand(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL, device=device)
    k_test = torch.rand(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL, device=device)
    v_test = torch.rand(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL, device=device)
    # 伪造 mask (batch_size, 1, seq_len_q, seq_len_k)
    # 对于自注意力, seq_len_q = seq_len_k = SRC_SEQ_LEN
    # 这里用 src_mask_example, 但需要调整形状
    # src_mask_example 形状是 (BATCH_SIZE, 1, 1, SRC_SEQ_LEN)
    # MultiHeadAttention 内部的 scaled_dot_product_attention 期望 mask 形状 (BATCH_SIZE, num_heads, seq_len_q, seq_len_k)
    # 或者 (BATCH_SIZE, 1, seq_len_q, seq_len_k)
    # 我们的 make_src_mask 产生的 (BATCH_SIZE, 1, 1, SRC_SEQ_LEN) 会被广播到 (BATCH_SIZE, 1, SRC_SEQ_LEN, SRC_SEQ_LEN)
    # 在 scaled_dot_product_attention 中, attn_scores 形状 (batch_size, num_heads, seq_len_q, seq_len_k)
    # mask 会被广播以匹配它。
    # 所以 (BATCH_SIZE, 1, 1, SRC_SEQ_LEN) 是可以的, 它会屏蔽 Key 中的 padding token。
    attn_out, attn_weights = multi_head_attn(q_test, k_test, v_test, mask=src_mask_example)
    print(f"MultiHeadAttention 输出形状: {attn_out.shape}") # (BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    print(f"MultiHeadAttention 权重形状: {attn_weights.shape}") # (BATCH_SIZE, NUM_HEADS, SRC_SEQ_LEN, SRC_SEQ_LEN)

    # --- 测试 PositionalEncoding ---
    print("\n--- 测试 PositionalEncoding ---")
    pos_encoder_test = PositionalEncoding(d_model=D_MODEL, max_len=MAX_SEQ_LEN).to(device)
    dummy_embeddings = torch.zeros(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL, device=device)
    pos_encoded_output = pos_encoder_test(dummy_embeddings)
    print(f"PositionalEncoding 输出形状: {pos_encoded_output.shape}") # (BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)
    # 可以尝试可视化位置编码 (例如第一个样本, 第一个维度)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.pcolormesh(pos_encoder_test.pe[0].cpu().numpy(), cmap='viridis')
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.colorbar()
    plt.title("Positional Encoding Matrix")
    plt.savefig("position.png")
    plt.show()

