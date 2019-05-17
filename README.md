# KerasLearning

## 1、核心层（各层函数只介绍一些常用参数，详细参数介绍可查阅Keras文档）

### 1.1全连接层：
+ 神经网络中最常用到的，实现对神经网络里的神经元激活。

    Dense（units, activation=’relu’, use_bias=True）

    参数说明：

        units: 全连接层输出的维度，即下一层神经元的个数。
    
        activation：激活函数，默认使用Relu。
    
        use_bias：是否使用bias偏置项。

 

### 1.2激活层：
+ 对上一层的输出应用激活函数。

    Activation(activation)

    参数说明：

        Activation：想要使用的激活函数，如：’relu’、’tanh’、‘sigmoid’等。

 

### 1.3Dropout层：
+ 对上一层的神经元随机选取一定比例的失活，不更新，但是权重仍然保留，防止过拟合。

    Dropout(rate)

    参数说明:

        rate：失活的比例，0-1的浮点数。

 

### 1.4Flatten层：
+ 将一个维度大于或等于3的高维矩阵，“压扁”为一个二维矩阵。即保留第一个维度（如：batch的个数），然后将剩下维度的值相乘作为“压扁”矩阵的第二个维度。

    Flatten()

 

### 1.5Reshape层：
+ 该层的作用和reshape一样，就是将输入的维度重构成特定的shape。

    Reshape(target_shape)

    参数说明：

        target_shape：目标矩阵的维度，不包含batch样本数。

    
    如我们想要一个9个元素的输入向量重构成一个(None, 3, 3)的二维矩阵：

    Reshape((3,3), input_length=(16, ))

 

### 1.6卷积层：
+ 卷积操作分为一维、二维、三维，分别为Conv1D、Conv2D、Conv3D。一维卷积主要应用于以时间序列数据或文本数据，二维卷积通常应用于图像数据。由于这三种的使用和参数都基本相同，所以主要以处理图像数据的Conv2D进行说明。

    Conv2D(filters, kernel_size, strides=(1, 1), padding=’valid’)

    参数说明：

        filters：卷积核的个数, 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
    
        kernel_size：卷积核的大小。
    
        strdes：步长，二维中默认为(1, 1)，一维默认为1。
    
        Padding：补“0”策略，’valid‘指卷积后的大小与原来的大小可以不同，’same‘则卷积后大小与原来大小一致。

 

### 1.7池化层：
+ 与卷积层一样，最大统计量池化和平均统计量池化也有三种，分别为MaxPooling1D、MaxPooling2D、MaxPooling3D和AveragePooling1D、AveragePooling2D、AveragePooling3D，由于使用和参数基本相同，所以主要以MaxPooling2D进行说明。

    MaxPooling(pool_size=(2,2), strides=None, padding=’valid’)

    参数说明：

        pool_size：长度为2的整数tuple，表示在横向和纵向的下采样样子，一维则为纵向的下采样因子。
    
        padding：和卷积层的padding一样。

 

### 1.8循环层：
+ 循环神经网络中的RNN、LSTM和GRU都继承本层，所以该父类的参数同样使用于对应的子类SimpleRNN、LSTM和GRU。

    Recurrent(return_sequences=False)

    参数说明：

        return_sequences：控制返回的类型，“False”返回输出序列的最后一个输出，“True”则返回整个序列。当我们要搭建多层神经网络（如深层LSTM）时，若不是最后一层，则需要将该参数设为True。

 

### 1.9嵌入层：
+ 该层只能用在模型的第一层，是将所有索引标号的稀疏矩阵映射到致密的低维矩阵。如我们对文本数据进行处理时，我们对每个词编号后，我们希望将词编号变成词向量就可以使用嵌入层。

    Embedding(input_dim, output_dim, input_length)

    参数说明：

        Input_dim：大于或等于0的整数，字典的长度即输入数据的个数。
    
        output_dim：输出的维度，如词向量的维度。
    
        input_length：当输入序列的长度为固定时为该长度，然后要在该层后加上Flatten层，然后再加上Dense层，则必须指定该参数，否则Dense层无法自动推断输出的维度。

    
    该层可能有点费解，举个例子，当我们有一个文本，该文本有100句话，我们已经通过一系列操作，使得文本变成一个(100,32)矩阵，每行代表一句话，每个元素代表一个词，我们希望将该词变为64维的词向量：

    Embedding(100, 64, input_length=32)

    则输出的矩阵的shape变为(100, 32, 64)：即每个词已经变成一个64维的词向量。

