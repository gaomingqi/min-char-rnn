"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
改编自 Andrej Karpathy 的 Min-Char-RNN
BSD License
"""
import numpy as np
import time
import matplotlib.pyplot as plt

# data I/O
# 读取训练数据(一个文本txt文件)
data = open('input.txt', 'r').read()    # should be simple plain text file
chars = sorted(list(set(data)))         # 找出构成文本的字符列表(报告中的 C )
                                        # 加排序的原因是, 确保每次运行时字符列表的顺序是相同的

data_size, vocab_size = len(data), len(chars)   # 计算文本长度, 与字符列表长度
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }  # 建立 字符 -> 索引 的字典, 用于生成独热码
ix_to_char = { i:ch for i,ch in enumerate(chars) }  # 建立 索引 -> 字符 的字典, 用于从独热码中恢复

# hyperparameters
# 模型的超参数设置
hidden_size = 100 # size of hidden layer of neurons, 隐藏态神经元数量
seq_length = 25 # number of steps to unroll the RNN for, 反向传播时, 模型只对最近25个时间步的参数做更新
learning_rate = 1e-1    # 学习率
epochs = 2500     # 对训练数据的迭代次数

class Min_char_rnn:
    def __init__(self, initType = None):
        # model parameters
        # 模型的参数设置
        if initType is None:
            self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01     # input to hidden, 输入(xt) -> 隐藏态(ht)映射的权重矩阵
            self.Whh = np.random.randn(hidden_size, hidden_size)*0.01    # hidden to hidden, 隐藏态(ht-1) -> 隐藏态(ht)映射的权重矩阵
            self.Why = np.random.randn(vocab_size, hidden_size)*0.01     # hidden to output, 隐藏态(ht) -> 输出(yt)映射的权重矩阵
            self.bh = np.zeros((hidden_size, 1))                         # hidden bias, 计算隐藏态时使用的偏移量
            self.by = np.zeros((vocab_size, 1))                          # output bias, 计算输出时使用的偏移量
        else:
            self.Whh = np.load('model/Whh.npy')
            self.Wxh = np.load('model/Wxh.npy')
            self.Why = np.load('model/Why.npy')
            self.bh = np.load('model/bh.npy')
            self.by = np.load('model/by.npy')

    def lossFun(self, inputs, targets, hprev):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state

        计算模型对当前输入的损失函数, 以及损失函数关于可学习参数的梯度值
        其中, inputs 与 targets 是两个长度为 25 的字符索引序列, 分别表示 25 个输入字符序列, 与对应的 25 个后续的真实字符列表
        hprev 表示上一次迭代过程生成的隐藏态

        """
        xs, hs, ys, ps = {}, {}, {}, {} # 分别表示输入数据、隐藏态、输出数据与字符概率
        hs[-1] = np.copy(hprev)         # 初始化隐藏态为上次训练迭代时生成的隐藏态
        loss = 0
        # forward pass
        # 前向传播过程
        for t in range(len(inputs)):    # 对 25 个字符做连续的预测过程, 同时累计 loss, 记录每个时步下生成的隐藏态与输出数据
            xs[t] = np.zeros((vocab_size,1))    # encode in 1-of-k representation
            xs[t][inputs[t]] = 1                # 对当前字符做独热码编码
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
            loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
        # backward pass: compute gradients going backwards
        # 初始化各个参数的梯度
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(hs[0])   # 报告中图 8 的 dLoss_t/dh_(t-1), 当前时步的损失函数关于上个时步隐藏态的梯度

        # 反向传播过程, 关于循环中涉及的梯度推导
        # 请查看报告的图 8
        for t in reversed(range(len(inputs))):
            dy = np.copy(ps[t])
            dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            # dy = dLoss/dy = Softmax 损失函数的梯度, 为何 -1 请参照报告的公式 5
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
            dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
            # 关于 tanh 梯度请参照报告的公式 4
            dbh += dhraw
            dWxh += np.dot(dhraw, xs[t].T)
            dWhh += np.dot(dhraw, hs[t-1].T)
            dhnext = np.dot(self.Whh.T, dhraw)

        # 对各个参数的梯度值做截断处理, 防止梯度爆炸
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
        return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

    def sample(self, h, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step

        对给定输入字符, 预测其后长度为 n 的字符序列
        其中, seed_ix 表示输入字符在字符集合的索引, h 表示隐藏态
        """
        x = np.zeros((vocab_size, 1))
        x[seed_ix] = 1                                              # 为输入字符做独热码编码
        ixes = []                                                   # 初始化输出字符序列
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)       # 对应报告中 公式(3) 的上半部分
            y = np.dot(self.Why, h) + self.by                                 # 对应报告中 公式(3) 的下半部分
            p = np.exp(y) / np.sum(np.exp(y))                       # 根据输出值计算 Softmax 概率
            ix = np.random.choice(range(vocab_size), p=p.ravel())   # 根据得到的字符概率分布做采样, 不是直接取最大概率对应字符
                                                                    # 使模型在相同输入下, 能够产生多样且合理的输出序列
            x = np.zeros((vocab_size, 1))
            x[ix] = 1                                               # 为输出字符做独热码编码, 作为下一时间步的输入
            ixes.append(ix)                                         # 将输出字符放入字符序列内, 用于返回预测结果
        return ixes

    def train(self):
        """
        模型训练函数, 对input.txt的完整字符序列进行 epochs 次数的训练
        :return: 无返回值, 训练结束会将 5 个模型参数保存在 npy 文件中
        """
        e, n, p = 0, 0, 0       # e: 控制训练进度, n: 控制显示进度, p: 字符指针
        trainData = []          # 保存训练数据 (epoch, iteration, loss)

        startTime = time.time()
        # 初始化 adagrad 更新参数, 即梯度缓存, 用于减缓垂直方向与水平方向梯度量的差别
        mWxh, mWhh, mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        mbh, mby = np.zeros_like(self.bh), np.zeros_like(self.by) # memory variables for Adagrad
        smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0, 使 loss 变化更加平缓
        hprev = np.zeros((hidden_size, 1))  # reset RNN memory

        # 训练迭代过程, 在经过 epochs 次对训练集的字符序列训练后, 跳出循环, 训练结束
        while True:
            # prepare inputs (we're sweeping from left to right in steps seq_length long)
            # 如果更新字符指针会越界, 就不管后面的数据了, 开始又一次对训练数据集的迭代
            if p+seq_length+1 >= len(data) or n == 0:
                hprev = np.zeros((hidden_size,1)) # reset RNN memory
                p = 0 # go from start of data
                e += 1
                if e > epochs:  # 如果对训练集的训练次数超过 epochs, 就结束训练过程
                    # 保存训练好的参数到 model 文件夹
                    np.save('model/Whh.npy', np.array(self.Whh))
                    np.save('model/Wxh.npy', np.array(self.Wxh))
                    np.save('model/Why.npy', np.array(self.Why))
                    np.save('model/bh.npy', np.array(self.bh))
                    np.save('model/by.npy', np.array(self.by))
                    np.save('TRAIN_DATA.npy', np.array(trainData))
                    break

            # 获取输入数据与 ground truth 数据
            inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
            targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

            # sample from the model now and then, 每100次迭代过程对模型做一次测试, 返回输入字符后200个字符的序列
            if n % 100 == 0:
                sample_ix = self.sample(hprev, inputs[0], 200)
                txt = ''.join(ix_to_char[ix] for ix in sample_ix)
                print('----\n %s \n----' % (txt, ))

            # forward seq_length characters through the net and fetch gradient
            # 获取当前迭代的损失函数值, 以及参数的梯度值, 每100次迭代显示一次
            loss, dWxh, dWhh, dWhy, dbh, dby, hprev = self.lossFun(inputs, targets, hprev)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            if n % 100 == 0:
                print('iter %d, loss: %f' % (n, smooth_loss)) # print progress
                trainData.append((e, n, smooth_loss))
                # 生成报告中分段的测试结果
                if n == 0 or n == 1000 or n == 10000 or n == 25000 or n == 50000 or n == 77400:
                    self.test('I', 50)

            # perform parameter update with Adagrad, 对模型的可学习参数, 使用 adagrad 方法更新
            for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                                        [dWxh, dWhh, dWhy, dbh, dby],
                                        [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

            p += seq_length # move data pointer, 更新字符指针
            n += 1 # iteration counter, 更新字符迭代次数

        endTime = time.time()
        print('训练结束, 使用时间：%d s' % (endTime - startTime))

    def test(self, seed=None, test_length=200):
        """
        对输入字符 seed, 返回其后续的, 连续 test_length 个预测字符
        """
        seed_ix = char_to_ix[seed]          # 获取输入字符在字符集的索引
        hprev = np.zeros((hidden_size, 1))  # 初始化隐藏态为 0
        sample_ix = self.sample(hprev, seed_ix, test_length)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)       # 将结果索引转换回字符, 并打印
        print('----\n %s \n----' % (txt,))

def linechart():
    """
    绘制损失函数在训练过程中变化的线图
    """
    trainData = np.load('TRAIN_DATA.npy')   # 读取训练中存储的元数据

    iteration = trainData[:, 1]
    loss = trainData[:, 2]

    plt.plot(iteration, loss, color='r', linewidth=2, label='length(hs)=100')

    plt.legend()                # 让图例生效
    plt.margins(0)
    plt.xlabel('Iterations')    # X轴标签
    plt.ylabel('Loss')          # Y轴标签

    plt.savefig('loss_changes.eps', dpi=1000, bbox_inches='tight')
    plt.show()


# 初始化模型对象，如果要从0开始训练模型，请在此处去掉参数
model = Min_char_rnn(1)

# model.train() # 训练模型

model.test(seed='P', test_length=200)     # 测试模型, 给出一个字符c, 预测长度为test_length的后续字符序列
                                    # 如不提供c, 则在训练集中随机取一个字符做测试
# linechart()

