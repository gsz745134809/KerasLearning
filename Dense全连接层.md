# Dense
全连接层

~~~python
keras.layers.Dense(units, activation=None, use_bias=True, 
                   kernel_initializer='glorot_uniform', bias_initializer='zeros', 
                   kernel_regularizer=None, bias_regularizer=None, 
                   activity_regularizer=None, kernel_constraint=None, 
                   bias_constraint=None)
~~~

```Dense``` 实现以下操作： ```output = activation(dot(input, kernel) + bias)``` 其中 ```activation``` 是按逐个元素计算的激活函数，```kernel``` 是由网络层创建的权值矩阵，以及 ```bias``` 是其创建的偏置向量 (只在 ```use_bias``` 为 ```True``` 时才有用)。

+ **注意**: 如果该层的输入的秩大于2，那么它首先被展平然后 再计算与 ```kernel``` 的点乘。

### 例

~~~python
# 作为 Sequential 模型的第一层
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
# 现在模型就会以尺寸为 (*, 16) 的数组作为输入，
# 其输出数组的尺寸为 (*, 32)

# 在第一层之后，你就不再需要指定输入的尺寸了：
model.add(Dense(32))

~~~

### 参数

+ **units**: 正整数，输出空间维度。
+ **activation**: 激活函数 (详见 [activations](https://keras.io/zh/activations/))。 若不指定，则不使用激活函数 (即，「线性」激活: ```a(x) = x```)。
+ **use_bias**: 布尔值，该层是否使用偏置向量。
+ **kernel_initializer**: ```kernel``` 权值矩阵的初始化器 (详见 [initializers](https://keras.io/zh/initializers/))。
+ **bias_initializer**: 偏置向量的初始化器 (see initializers).
+ **kernel_regularizer**: 运用到 ```kernel``` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/)。
+ **bias_regularizer**: 运用到偏置向的的正则化函数 (详见 regularizer)。
+ **activity_regularizer**: 运用到层的输出的正则化函数 (它的 "activation")。 (详见 regularizer)。
+ **kernel_constraint**: 运用到 ```kernel``` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/)。
+ **bias_constraint**: 运用到偏置向量的约束函数 (详见 constraints)。

### 输入尺寸

nD 张量，尺寸: ```(batch_size, ..., input_dim)```。 最常见的情况是一个尺寸为 ```(batch_size, input_dim)``` 的 2D 输入。

### 输出尺寸

nD 张量，尺寸: ```(batch_size, ..., units)```。 例如，对于尺寸为 ```(batch_size, input_dim)``` 的 2D 输入， 输出的尺寸为 ```(batch_size, units)```。



~~~python
# Dense层源码
class Dense(Layer):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    # Output shape
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    @interfaces.legacy_dense_support
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

~~~

