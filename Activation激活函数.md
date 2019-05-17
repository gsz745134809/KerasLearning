# 激活函数的用法

激活函数可以通过设置单独的激活层实现，也可以在构造层对象时通过传递 ```activation``` 参数实现：

~~~python
from keras.layers import Activation, Dense

model.add(Dense(64))
model.add(Activation('tanh'))
~~~
等价于：

~~~python 
model.add(Dense(64, activation='tanh'))
~~~
你也可以通过传递一个逐元素运算的 Theano/TensorFlow/CNTK 函数来作为激活函数：

~~~python
from keras import backend as K

model.add(Dense(64, activation=K.tanh))
model.add(Activation(K.tanh))
~~~









