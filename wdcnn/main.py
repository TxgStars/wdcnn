import torch
import preprocess
from torch import nn
import train as tr
#训练参数
batch_size = 128
epochs = 20
num_classer = 10
length = 2048 #采样长度
BatchNorm = True #是否批量归一化
number = 1000 #每类样本数量
normal = True #是否对样本标准化
rate = [0.7, 0.2, 0.1] #训练集、测试集、样本集划分比例

path = "data\\0HP"
train_X, train_Y, test_X, test_Y, valid_X, valid_Y = preprocess.prepro(data_path=path,
                                                                       length=length,
                                                                       number=number,
                                                                       slice_rate=rate,
                                                                       enc=True,
                                                                       enc_step=28,
                                                                       normal=normal)

#把数据从array个数转成tensor格式
#把列信号看作一个length * 1的二维“图片”，然后就可以用卷积了
#但是torch框架里有一维卷积，所有不需要使用二维卷积来实现
#一维卷积里数据只需要三个维度，（样本数/batchsize， 输入通道数， 样本长度）
train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).float() #将onehot标签数据类型转换成torch.float类型，以避免在计算loss时因为数据类型不匹配报错
test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).float()
valid_X = torch.from_numpy(valid_X).float()
valid_Y = torch.from_numpy(valid_Y).float()
#将样本数据增加维度变成三维张量，torch里的一维卷积是对三维张量进行
train_X = train_X[:, None, :]
test_X = test_X[:, None, :]
valid_X = valid_X[:, None, :]

#将数据切分为随机batch_size
train_iter = tr.slice_to_batch_size(train_X, train_Y, batch_size=batch_size, shuffle=True)
test_iter = tr.slice_to_batch_size(test_X, test_Y, batch_size=batch_size, shuffle=True)

#定义模型
net = nn.Sequential(
        #第一层卷积&池化,输出：（None, 16, 64）
        nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24), nn.BatchNorm1d(16), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        #第二层卷积&池化，输出（None, 32,32）
        nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        #第三层卷积&池化，输出（None，64，16）
        nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        #第四层卷积&池化，输出（None, 64, 8）
        nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        #第五层卷积&池化，输出（None, 64, 3）
        nn.Conv1d(64, 64, kernel_size=3), nn.BatchNorm1d(64), nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        #将最后一层池化的结果展平，形状是（None, 192）
        nn.Flatten(),
        nn.Linear(192, 100), nn.ReLU(),
        nn.Linear(100, 10)
        #nn.Softmax(dim=1) 使用torch里面的交叉熵计算loss时会自动给预测值添加softmax，因此这里不需要softmax层了
)

#检查模型
"""
X = torch.randn(1, 1, 2048)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
"""

tr.train(net, train_iter, test_iter, num_epochs=epochs, lr=0.01, device=tr.try_gpu())

