import torch
import preprocess
import numpy as np
from torch import nn
import torch.utils.data as Data
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display

def slice_to_batch_size(data_X, data_Y, batch_size, shuffle=True):
    """
    将数据切分成batcha_size
    :param data_X: 样本数据集
    :param data_Y: 样本标签
    :batcha_size: batch_size的大小
    :param shuffle: 是否随机划分
    :return: data_iter 迭代器
    """
    torch_dataset = Data.TensorDataset(data_X, data_Y)
    data_iter = Data.DataLoader(dataset=torch_dataset, batch_size=batch_size, shuffle=True)
    return data_iter

def try_gpu(i=0):
    """如果存在GPU，返回将torch框架与相应GPU关联起来的对象，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def accuracy(y_hat, y):
    """计算一个batch的精度"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #获取预测值最大的标签所在的位置
    y_true = y.clone() #必须克隆，否则使用y = y.argmax会修改传入的原值，因为python参数传递是引用
    y_true = y_true.argmax(axis=1) #在onehot标签中，最大值所在的位置就是标签1所在位置
    cmp = y_hat.type(y_true.dtype) == y_true #与真实标签对比
    return float(cmp.type(y.dtype).sum()) #返回预测精度

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """计算测试精度，使用gpu"""
    if isinstance(net, nn.Module):
        net.eval() #设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), len(y))
    return metric[0] / metric[1]

class Accumulator:
    """对n个变量进行累加"""
    def __init__(self, n):
        """n表示要累加的变量的个数"""
        self.data = [0.0] * n #为n个变量创建列表，数据类型为float32
    def add(self, *args):
        """
        :param args: 允许传入多个参数，将传入的多个参数作为元组，传入的参数个数必须与data中的变量个数n相同
        args中就保存了新的需要增加的变量的值
        :return:没有返回值
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)] #a是原来变量的旧值，b是需要增加的值，a + b就得到增加后总的值

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """这个函数的作用是实现该类的对象能够使用索引访问数据
            也就是对这个类的对象使用索引的方式时就会调用这个函数，索引值作为参数idx"""
        return self.data[idx]

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置图片参数"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """动态绘制训练曲线"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        #采用增量式绘制曲线
        if legend is None:
            legend=[]
        backend_inline.set_matplotlib_formats('svg') #设置SVG形式显示曲线
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        #使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim,
                                            xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts #X记录横轴数据，Y记录纵轴数据，fmts用于设置线的颜色

    def add(self, x, y):
        """向图表中添加数据点，x是横轴点，y是纵轴点"""
        if not hasattr(y, "__len__"):
            y = [y] #如果不可迭代，那将其变为列表
        n = len(y) #一次只添加一组数据点，因此y有几个数就有几组数据
        if not hasattr(x, "__len__"):
            x = [x] * n #生成n个x元素的列表
        if not self.X:
            self.X = [[] for _ in range(n)] #创建n组横轴坐标，也就有几条线的数据
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.draw()
        plt.pause(0.001)
        display.display(self.fig)
        display.clear_output(wait=True)

    def show(self):
        display.display(self.fig)

def train(net, train_iter, test_iter, num_epochs, lr, device):
    """
    :param net: 使用的网络模型
    :param train_iter: 训练数据集，已经按照batch_size划分为一个迭代器
    :param test_iter:  测试数据集迭代器
    :param num_epochs: 训练周期
    :param lr: 训练速度
    :param device: 在哪个设备上训练
    :return:
    """
    def init_weights(m):
        """定义网络的参数初始化方法"""
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights) #对网络进行初始化
    print("training on ", device)
    net.to(device) #将模型参数移动到指定设备上
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) #设置优化器
    loss = nn.CrossEntropyLoss() #设置交叉熵误差,默认计算的是输入的一个batch的平均误差
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    num_batches = len(train_iter) #batch的个数
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train() #设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            optimizer.zero_grad() #梯度清零
            X, y = X.to(device), y.to(device) #将当前batch的数据传入GPU，并且属于for循环里的临时变量，因此每一次for结束都会被清除显存，也就是每次显卡上只有一个batch的数据，这样可以节省显存
            y_hat = net(X) #获取预测值
            l = loss(y_hat, y) #计算得到的是一个batch的平均误差,并且要求真实的one_hot标签的数据类型与y_hat相同，那也就意味着y要转成float类型
            l.backward()
            optimizer.step() #更新参数
            #计算当前训练下的训练精度
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) ==0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f},'
              f'test acc {test_acc:.3f}')
    plt.show()

