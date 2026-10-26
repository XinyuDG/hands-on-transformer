# transformer-

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337) # 随机种子 固定每次生成的随机数
import random
import textwrap # 自动换行

# file_name = "hong_lou_meng_short.txt"
file_name = "hong_lou_meng_80.txt"

#-----------------------超参数--------------------------
# 把所有常量定义到前面（超参数）
# block_size = 5  # 训练验证的字符串长度
# batch_size = 16 # 一次读取几个block（字符串片段）
block_size = 64  # 训练验证的字符串长度
batch_size = 256
device = "cuda" if torch.cuda.is_available() else "cpu" # 定义设备使用gpu
#device = torch.device("cpu")
wrap_width = 50 # 用来对生成的结果自动换行的（可以先不用管）
#size = 10 # 几个值需要做嵌入()
n_embedding = 384 # 嵌入后的维度
n_heads = 8
head_size = n_embedding // n_heads
#embedding_table = nn.Embedding(size, n_embedding)
learning_rate = 0.0005
torch.manual_seed(1337) # 随机种子
max_iters = 1000
eval_iters = 200 # 每评估一次loss用多少的iteration做平均值
eval_interval = int(max_iters / 10) #一共测10次loss
dropout_value = 0.2 #有20%的节点归零，不参与运算
n_layer = 6 # 六层block(6级残差网络)


# ---------------数据预处理--------------------------------
with open(file_name, 'r', encoding='gb18030') as f:
    text = f.read() # str

# 有序不重复的列表
chars = sorted(list(set(text))) #去除重复字然后转换成列表然后排序

vocab_size = len(chars) #打印词汇表长度

# 字符和整数之间的投影
stoi = {ch:i for i, ch in enumerate(chars)} # 符号到整数
itos = {i:ch for i, ch in enumerate(chars)} # 整数到符号

encode = lambda str1: [stoi[c] for c in str1] # 定义一个函数的简便方式，：前是输入，后是输出
encode_list = encode("早知他来，我就不来了") # 把字符串转化成数字串列表
decode = lambda list1: "".join([itos[i] for i in list1]) # 把数字串转化成字符串然后合并
data = torch.tensor(encode(text), dtype = torch.long) # 用整数表示字符。转化成pytorch数据张量

n = int(0.9*len(data)) #前90%用于训练
train_data = data[:n] # 训练数据
val_data = data[n:] # 验证数据
print(f"文件{file_name}读取完成")

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix  = torch.randint(len(data) - block_size, (batch_size, )) # 取的随机位置是起始点的位置，减去block_size的目的是保证每次能取到一个block_size。(假设random取到文本最后一个往后就不足一个block_size长度)
    # 第一个参数是取值范围，第二个参数是随机生成整数的形状
    x = torch.stack([data[i:i+block_size] for i in ix]) # 对ix的每个位置都取一个字符串
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # 把x向后移一个字符, 想要的输出值（target)

    x, y = x.to(device), y.to(device)

    # token_list_x = x.tolist() # 转化成列表
    # token_list_y = y.tolist()
    # for str_list in token_list_x:
    #     print(decode(str_list)) # 变回字符
    # for str_list in token_list_y:
    #     print(decode(str_list))

    return x, y

#---------------损失评测-------------------------------------（这部分不是重点可以简单看下）
@torch.no_grad() #不做梯度计算的decorator,作用域为整个函数
def estimate_loss(model):
    out = {}
#等价于
# def estimate_loss(model):
#     with torch.no_grad():
#     out = ()
    model.eval() # 把模型转化为evaluate模式（默认模式是train)
    for split in ['train','val']: # 训练组和测试组的损失都要计算一下
        losses = torch.zeros(eval_iters) # 建立一个初始值为0的容器，用于储存loss值
        for k in range(eval_iters): # 进行100次get_batch(每次取batch_size个)
            X, Y = get_batch(split) # split 是一个字符串，用来控制get_batch()函数的行为
            logits, loss = model(X,Y) # model的输入值一个是index(以每个字符的序号表示的序列)，一个是target
            losses[k] = loss.item()
        out[split] = losses.mean() # out是含有两个元素的字典，一个是train, 一个是val, 每个元素对应一个loss的平均值
    model.train() # 再转换为训练模式（如果之前没有转为evaluate模式，则不需要这一步，因为模型建立后默认为训练模式
    return out

# Head 类
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding, head_size, bias = False) # 线性变换层 会通过一个可学习的权重矩阵进行变换
        self.query = nn.Linear(n_embedding, head_size, bias = False) # 线性变换层
        self.value = nn.Linear(n_embedding, head_size, bias = False) # 线性变换层 （B, T, n_embedding) --->（B, T, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(block_size,block_size))) # register_buffer定义一个不可训练的结构（相当于常量）
        # torch.tril(torch.ones(block_size,block_size))对全是1的矩阵做三角处理变成一个(下）三角矩阵
        # 后面可用self.tril调用
        self.dropout = nn.Dropout(dropout_value) #训练中，一部分随机节点会设置为0，以减少过拟合增强模型的通用性


    def forward(self, x):
        B, T, C = x.shape
        K = self.key(x)
        Q = self.query(x)
        #wei = torch.ones(T, T) # 注意力方阵 (B, T, T)
        wei = Q @ K.transpose(-2,-1) * (K.shape[-1] ** -0.5) # (B, T, head_size) 对倒数第二个和倒数第一个维度（T，head_size)做转置，第一个维度batch不变（每个batch分别计算）
        # score / head_size的开方。K.shape[-1] ** -0.5就是head_size的开方。
        # 如果head_size过大的话，点乘得到的score就会大（Q,K的第三个维度是head_size)，因此为了去除head_size的影响除以head_size的开方
        wei = wei.masked_fill(self.tril == 0, float("-inf")) # 掩码矩阵
        wei = F.softmax(wei, dim = -1) 
        wei = self.dropout(wei) # 随机去掉一些值（变成0），增加网络的准确性

        v = self.value(x) # 对每一个 token 的 embedding 向量，做线性变换，投影成一个新的向量 
        out = wei @ v # 矩阵乘法（B, T, head_size)
        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)]) # 实例化num_heads个注意力头然后放入ModuleList中
        self.proj = nn.Linear(head_size * n_heads, n_embedding)
        self.dropout = nn.Dropout(dropout_value)
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # 对self.heads列表中的每个 Head 模块都调用 forward(x), 把最后一个维度连接起来（cat的作用是连接）
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module): # 把中间的神经网络层统一放到这里方便复用
    def __init__(self, n_embedding):
        super().__init__()
        self.net = nn.Sequential( # Sequential会自动执行
            nn.Linear(n_embedding, n_embedding*4),
            nn.ReLU(),
            nn.Linear(n_embedding * 4, n_embedding),
            nn.Dropout(dropout_value)
        ) # 两层线性变换加一个激活函数
    def forward(self, x):
        return self.net(x)

class Block(nn.Module): # 残差网络
    def __init__(self, n_embedding, n_heads):
        super().__init__()
        self.sa = MultiHeadAttention(n_heads, head_size) # 自注意力（多头注意力）
        self.ffwd = FeedForward(n_embedding) 
        self.ln1 = nn.LayerNorm(n_embedding) # 正则化
        self.ln2 = nn.LayerNorm(n_embedding)

    def forward(self, x):
        # 残差多头注意力网络
        x = x + self.sa(self.ln1(x)) # 先做一个正则化（减少训练时的梯度消失/爆炸，使模型更容易训练、更快收敛，提高模型稳定性）
        # 残差线性前馈层
        x = x + self.ffwd(self.ln2(x)) # 再做一个正则化
        return x


#---------------1. 傻瓜模型(还没有注意力机制) 2.语言模型-------------------------------
class LanguageModel(nn.Module): # nn.Module是pytorch现有的神经网络的类
    def __init__(self):
        super().__init__()
        print(f"block_size:{block_size}")
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding) # 把vocab_size中的每个token变成n_embedding的长度
        self.position_embedding_table = nn.Embedding(block_size, n_embedding) # 位置编码（在这个block内的位置）
        # 加入head
        self.multi_head = MultiHeadAttention(n_heads, head_size)
        #self.network1 = nn.Linear(n_embedding, n_embedding*4) # 全连接神经网络 （输入值是n_embedding,输出是vocab_size)
        # nn.Linear(in_features, out_features) 只会作用于输入张量的最后一维**，不管前面有多少维度，它都一视同仁地当作“批次”处理。**
        #self.network2 = nn.Linear(n_embedding*4, vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embedding, n_heads) for _ in range(n_layer)]) # 多级残差多头注意力
        # *的作用：把列表 [Block(...), Block(...), ...] 拆成多个参数传进去
        #调用会自动执行：
            # x1 = Block1(x)
            # x2 = Block2(x1)
            # x3 = Block3(x2)
        # final layer norm
        self.ln_f = nn.LayerNorm(n_embedding) # 最后一层
        self.ln_head =  nn.Linear(n_embedding, vocab_size) # 最后转换成vocab_size长度的token以进行softmax
        

    def forward(self, idx, targets = None):
        B, T = idx.shape # B是batch_size, T是block_size, 数据是token（整数）形式
        token_embd = self.token_embedding_table(idx)
        position_idx = torch.arange(T, device=idx.device) # 从0到T的数 生成tensor[0,1,2,3,...,T]
        # print(position_idx.shape)
        position_embd = self.position_embedding_table(position_idx) # 把位置进行编码
        x = token_embd + position_embd # (B,T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.ln_head(x)
        # 加入head
        # head_out = self.multi_head(x)
        # logits = torch.relu(self.network1(head_out)) # (B,T vocab_size) 就是这里在预测， 将模型处理过后的隐藏状态（embedding 表示）映射到词表大小（vocab_size）的向量空间
        # # 加入 relu 激活函数引入非线性
        # logits = self.network2(logits)

        if targets is None: #
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C) # 摊平 （把两个维度摊平到一个维度）比如把两个batch 拼成一个batch
            targets = targets.view(B*T) # 二阶变一阶
            loss = F.cross_entropy(logits, targets) # 交叉熵

        #random_tensor = torch.rand(B, T, vocab_size)
        #logits = random_tensor / random_tensor.sum(dim = -1, keepdim = True) #这里dim = -1 等价于dim = 2，表示vocab_size这个维度
        #loss = None
        return logits, loss

    def generate(self, token_sequ, max_new_tokens): # token_sequ是已知的上文， max_new_tokens是续写的长度
        for _ in range(max_new_tokens): # 用_表示循环的量再循环里面没用
            tokens_input = token_sequ[:, -block_size:] # -block_size: 表示从右往左数，取最后 block_size 个 token (用来预测下一个字)
            logits, loss = self.forward(tokens_input)  # logits原来的形状是（B,T,vocab_size)
            logits = logits[:, -1, :] # 只取字符串最后一个(也就是block中的最后一个向量），（概率分布的向量格式）
            probs = F.softmax(logits, dim = -1) # logits.shape = (B, 1, vocab_size)
            #print("token_sequ device:", token_sequ.device)
            token_next = torch.multinomial(probs, num_samples = 1) # 把概览分布向量变成one hot向量，torch.multinomial(...)：表示“按照这些概率抽样”
            token_next = token_next.to(token_sequ.device)
            #print("token_next device:", token_next.device)
            # 概率分布向量 --> one-hot向量 --> 整数token
            

            token_sequ = torch.cat((token_sequ, token_next), dim = 1)
        new_tokens = token_sequ[:, -max_new_tokens:] #只取新生成的
        return new_tokens
    

#x,y = get_batch("train")

#--------------------------主函数------------------------------
def main(): # 主运行函数
    print(f"训练内容：{file_name}")
    model = LanguageModel() # 实例化 （此时模型的参数初始化了）
    model = model.to(device)
    #out = model(x) # 调用forward函数（只有forward函数可以因为这是在nn.Module父类里定义好的可以隐式调用）
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters') # 打印有多少个参数

    #设定一个优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    # 训练循环
    for i in range(max_iters):

        if i % eval_interval == 0 or i == max_iters - 1: # 每eval_interval间隔打印Loss或者是最后一次
            losses = estimate_loss(model)
            print(f"step{i}: train loss {losses['train']:.4f}.val loss {losses['val']:.4f}")

        # 取样
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb) # 前馈运算 （预测并计算损失）
        optimizer.zero_grad(set_to_none=True) # 把模型中原来的梯度归零
        loss.backward() # 反向传播，计算新的梯度
        optimizer.step() # 做一步优化计算

    print("训练阶数，下面开始生成内容")

    #-----------------------测试generate函数----------------------------------
    max_new_tokens = 500
    start_idx = random.randint(0, len(val_data)-block_size-max_new_tokens)

    # 上文内容
    # 先初始化为0然后再赋值
    context = torch.zeros((1, block_size), dtype = torch.long, device=device) # (B, T) B = 1, T = block_size
    context[0, :] = val_data[start_idx:start_idx+block_size] # 输出是一个block
    context_str = decode(context[0].tolist()) # 把token列表变成字符串 （一阶张量），此时还没进行embedding所以content是一个2d张量
    wrapped_context_str = textwrap.fill(context_str, width = wrap_width) # 自动换行

    #真实下文
    real_next_tokens = torch.zeros((1, max_new_tokens), dtype = torch.long, device=device)
    real_next_tokens[0, :] = val_data[start_idx+block_size:start_idx+block_size+max_new_tokens] # 上文结束的点是下文开始的点
    real_next_tokens_str = decode(real_next_tokens[0].tolist()) # 转成字符串
    wrapped_real_next_tokens_str = textwrap.fill(real_next_tokens_str, width = wrap_width)

    print("context device:", context.device)


    #生成下文
    generated_tokens = model.generate(context, max_new_tokens)
    generated_str = decode(generated_tokens[0].tolist()) # 转成字符串
    wrapped_generated_str = textwrap.fill(generated_str, width = wrap_width)

    print("上文内容：")
    print(wrapped_context_str)
    print()
    print("生成内容：")
    print(wrapped_generated_str)
    print()
    print("真实下文内容：")
    print(wrapped_real_next_tokens_str)

#------------运行---------------
main()
# head = Head(5)
# print(f"三角矩阵trill")
# print(head.tril)

# wei = torch.ones(block_size, block_size) # 注意力矩阵
# print("注意力矩阵")
# print(wei)

# # 做掩码处理
# wei = wei.masked_fill(head.tril == 0, float("-inf")) # 对于head中等于0的值，把注意力矩阵中对应位置的值改为负无穷大，其余不变