# wdcnn
实现的是WDCNN的pytorch版本代码，对应论文的第三章 data包含了四个数据文件夹，这里只使用了0HP文件夹中的数据，里面包含了正常、内圈、外圈、滚动体共10种状态 preprocess.py的功能是对数据进行采样、编码，虽然划分出来了验证集但是并没有使用 train.py定义了用于模型训练以及显示的函数和类 main.py定义了网络模型，调用另外两个py文件获取数据和进行训练 直接运行main.py文件即可得到的结果