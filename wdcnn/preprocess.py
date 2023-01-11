import os

import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from sklearn import preprocessing  # 0-1编码
from sklearn.preprocessing import StandardScaler #归一化标准工具
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分数据集

def catch_into_dict(data_path):
    """读取指定路径下的mat文件并保存为字典
        data_path  文件路径，可以是多个文件
        返回值为data,格式 字典：
        data的键名key 为对应读取的文件名
        data的键值value 对应的振动振动信号数据"""
    filenames = os.listdir(data_path) # 获取指定文件路径下所有mat文件名，包括扩展名，返回的是一个包含所有文件名的列表
    data = {} #创建一个空字典来存储读取到的数据
    for f in filenames:
        filePath = os.path.join(data_path, f) #获取目标文件的路径，即data_path + f
        file = io.loadmat(filePath) #读取mat文件，返回一个字典，需要根据key来选择相应的数据进行保存
        keys = file.keys() #获取所读取到的字典的所有键名
        for key in keys:
            if 'DE' in key: #如果key中存在子字符串DE，那就是所需要的数据，把相应的数据保存到data中
                data[f] = file[key].ravel() #ravel作用是将多维数组展开成一维，这里的目的是把维度数据为1的维度取消掉
                """
                显示原始数据的图像
                print(data[f].shape)
                plt.figure()
                X = np.arange(0, 1024)
                plt.plot(X, np.array(file[key].ravel())[0:1024])
                plt.show()
                """
    return data

def slice_data(data, length=864, number=1000, slice_rate=[0.7, 0.2, 0.1], enc=True, enc_step=28):
    """将数据切片成一个个样本
        返回值说明
        train_samples 训练数据字典
        test_and_valid_samples 测试集和验证集数据字典（后续再把它们随机分开）
        参数说明
        length 每条样本的切片长度
        number 采集的样本数，默认1000
        slicer_rate 训练集、测试集、验证集的比例
        enc 是否采用数据增强
        enc_step 数据增强采集顺延间隔"""
    train_samples = {} #用于存放所有文件的切片结果，每个文件的样本对应字典中一个元素，作为训练集
    test_and_valid_samples = {} #用于存放训练集
    keys = data.keys() #获取字典的所有键值
    #通过遍历健实现遍历所有数据,先采集训练样本，后采集测试样本和验证样本
    for key in keys:
        train_data = [] #临时存储切片得到的训练样本
        total_length = len(data[key]) #数据的总长度
        end_index = int(total_length * slice_rate[0]) #采集训练样本时，只能采集到前面训练样本所占比例，相当于把整个数据集也分为了训练集、测试集、验证集,至于开始采样的位置由随机数确定
        train_num = int(number * slice_rate[0]) #获取训练样本总数量 一定要强制类型转换为整数，不然后面使用==判断就会出现问题
        if enc:
            enc_times = length // enc_step #在一个length里需要增强采集的次数
            steps = 0 #记录采样次数，采样到达train_num停止
            for j in range(train_num):
                label = False #标志位，用于确定是否停止采样
                start_index = np.random.randint(low=0, high=end_index - 2 * length) #随机产生采样起始点，减去2length是为了防止实际采样区间超出训练集范围，不能只减去1个length，要考虑增强采样的影响
                for h in range(enc_times):
                    temp = data[key][start_index:start_index + length] #获取切片样本
                    train_data.append(temp) #放入临时列表
                    steps += 1 #采样次数加1
                    start_index = start_index + enc_step #更新起始采样点
                    if steps == train_num:
                        label = True
                        break
                if label:
                    break
        else:
            for j in range(train_num):
                start_index = np.random.randint(low=0, high=end_index - 2 * length)
                temp = data[key][start_index: start_index + length]
                train_data.append(temp)
        train_samples[key] = np.array(train_data) #把list转换成array格式，是一个二维数组，（样本数，样本长度）然后把采样得到的数据放入训练字典中，
        #采集测试数据和验证数据
        test_and_valid_num = int(number * (1 - slice_rate[0])) #测试集和验证集的样本数
        test_and_valid_data = [] #临时存放列表，后面记得使用np.array转成numpy类型，以保证数据格式统一
        for j in range(test_and_valid_num):
            start_index = np.random.randint(low=end_index, high=total_length - length)
            temp = data[key][start_index: start_index + length]
            test_and_valid_data.append(temp)
        test_and_valid_samples[key] = np.array(test_and_valid_data) #将采集到的数据放入字典

    return train_samples, test_and_valid_samples

def add_label(data):
    """给数据划分标签，B007~B021，IR007~IR021, OR007~OR021, normal分别标号为 0~9
        后续再进行onehot编码
        参数说明
        data 传入的样本数据，格式为字典
        返回值说明
        标签值Y 元素范围为0~9分别代表10类
        样本集X 所有样本的数据 每个样本的位置和它在Y中的标签位置一一对应"""
    label = 0
    X = [] #空列表，用来存放所有的样本，最后返回时记得转换成array类型
    Y = [] #空列表，用来存放标签
    for key in data.keys():
        x_len = len(data[key]) #当前访问的数据类型的样本数
        y = [label] * x_len #生成标签，个数与当前访问的样本集中样本个数相同，并且每个样本对应一个标签,这里的标签是0，1，2...
        Y.append(y) #放入标签列表中
        X.append(data[key]) #把当前访问的样本集放入空列表
        label += 1 #下一次循环之前要给标签值加1
    #for循环结束后，X中按顺序存放了所有文件中的样本集，Y中存放了所有对应的标签值
    X = np.array(X) #此时X是三个维度，（样本类型数，单个类型的样本数，样本长度）
    X = X.reshape((-1, X.shape[-1])) #把维度变成（总样本数， 样本长度）
    return X, np.array(Y).ravel()

def one_hot(data_Y):
    """one_hot编码
        一共有10类，因此用10位0-1表示
        参数data_Y是标签，对应0~9
        返回值Y是生成的one_hot编码，格式是（样本数, 10）
        """
    Encoder = preprocessing.OneHotEncoder() #创建一个onthot编码对象，这个对象会存储如何实现onehot标签的信息
    data_Y = data_Y.reshape((-1, 1)) #把当前0，1，2...标签变成列向量
    Encoder.fit(data_Y) #根据data_Y里0，1，2...有多少个不同的数设置onehot标签模式,并且按照顺序，第一位二进制为1表示原来的标签0，第二位二进制数为1表示原来的标签1，依此类推
    Y = Encoder.transform(data_Y).toarray() #对data_Y应用Encoder对应的编码模式
    Y = np.asarray(Y, dtype=np.int32) #将Y转成array格式
    return Y

def slice_test_and_valid(te_va_X, te_va_Y, slice_rate=[0.7, 0.2, 0.1]):
    """将测试集和验证集拆开
        参数说明
        te_va_X 输入数据集
        te_va_Y 输入标签
        slice_rate 测试集、验证集分配比例
        返回值说明
        text_X 测试集
        test_Y 测试集标签
        valid_X 验证集
        valid_Y 验证集标签"""
    ss = StratifiedShuffleSplit(n_splits=1,
                                train_size=slice_rate[1] / (slice_rate[1] + slice_rate[2]))
    #ss是一个用于分组的对象，n_splits就表示分成多少组，train_size就表示训练集分多少比例，那剩下的就是测试集的，稍微利用一下就可以用来分测试集和验证集
    #ss.split返回的是第一个维度的分组结果索引，也就是样本序号数组,输入参数一定要包含标签
    for test_index, valid_index in ss.split(te_va_X, te_va_Y):
        test_X, valid_X = te_va_X[test_index], te_va_X[valid_index]
        test_Y, valid_Y = te_va_Y[test_index], te_va_Y[valid_index]
    return test_X, test_Y, valid_X, valid_Y

def prepro(data_path, length=864, number=1000, slice_rate=[0.7, 0.2, 0.1], enc=True, enc_step=28, normal=False):
    """
    :param data_path:文件路径，到达mat文件所在的文件夹层
    :param length:采样长度
    :param number: 每类数据采样次数
    :param slice_rate: 训练集、测试集、验证集划分比例
    :param enc: 是否进行数据增强
    :param enc_step: 数据增强步长
    :param normal: 是否对数据进行标准归一化
    :return: 返回训练集、测试集、验证集 train_X, test_X, valid_X,及其相应的one_hot标签train_Y, test_Y, valid_Y
    """
    data = catch_into_dict(data_path)  # 获取数据
    train_data, test_and_valid_data = slice_data(data, length=length, number=number, slice_rate=slice_rate,
                                                 enc=enc, enc_step=enc_step)  # 采集数据，将数据集切割成指定长度的样本
    train_X, train_Y = add_label(train_data)  # 为训练集制作标签
    test_and_valid_X, test_and_valid_Y = add_label(test_and_valid_data)  # 为测试集制作标签
    train_Y = one_hot(train_Y)  # 对训练集进行one_hot编码
    test_and_valid_Y = one_hot(test_and_valid_Y)
    # 进行归一化处理
    if normal:
        scaler1 = StandardScaler()
        train_X = scaler1.fit_transform(train_X)  # 实现归一化操作，这一行代码执行后，对象scaler1就包含了train_X的均值方差等信息
        scaler2 = StandardScaler()
        test_and_valid_X = scaler2.fit_transform(test_and_valid_X)
    test_X, text_Y, valid_X, valid_Y = slice_test_and_valid(test_and_valid_X, test_and_valid_Y, slice_rate=slice_rate) #划分测试集，验证集，在这个project中实际上并没有使用验证集
    return train_X, train_Y, test_X, text_Y, valid_X, valid_Y

if __name__ == "__main__":
    normal = True
    # 没有指定绝对路径，那么就是基于当前项目所在文件路径下的相对路径
    data_path = "data\\0HP"
    train_X, train_Y, test_X, test_Y, valid_X, valid_Y = prepro(data_path=data_path, length=864, number=1000,
                                                                slice_rate=[0.7, 0.2, 0.1],
                                                                enc=True,
                                                                enc_step=28,
                                                                normal=False)