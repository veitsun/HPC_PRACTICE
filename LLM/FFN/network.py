import numpy as np
# lets define a python class and write an init function where well spcify our parameters such as input ,hidden , output layers

class neural_network(object):
  def __init__(self):
    self.inputSize = 2
    self.outputSize = 1
    self.hiddenSize = 3
  
    # so far we have the data all set up
    # now we need to csee if we can predict a score for our input data 
    # 前向传播是我们的神经网络预测输入数据分数的方式
    # obj = neural_network()

    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (2x3) weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (3x1) weight matrix from hidden to output layer

  # 一旦设置好所有的变量，我们就可以编写 forward 传播函数了，
  # 激活函数
  def sigmod(sef, s):
    return 1/(1+np.exp(-s))

  # forward function 前向传播函数
  # 定义通过神经网络的前向传播函数。
  # 首先我们需要将输入与第一组随机权重的点积相乘。

  def forward(self, X):
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of 3x2 weights
    self.z2 = self.sigmod(self.z) # 激活函数
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of 3x1 weights
    o = self.sigmod(self.z3) # final activation function
    return o
  
  def sigmodPrime(self, s): # 激活函数的导数
    return s * (1 - s)

  # backward propagation 反向传播
  # 我们已经有了一系列的权重，我们需要改变这些权重来让我们的神经网络能够预测更准确的分数
  # 反向传播通过使用损失函数来计算网络与目标输出的距离
  def backward(self, X, y, o):
    # 首先，我们通过计算输出层o 和 实际值y 之间的差值来找到函数中的误差。


"""
我们将通过以下方式计算权重的增量变化：
  - 通过计算预测输出与实际输出 y 之间的差值，找到输出层的误差
  - 将我们的 S 型激活函数的导数应用于输出层误差。我们将此结果称为增来给你输出总和
  - 使用输出层误差的增量输出总和，通过与第二个权重矩阵进行点积计算出z2 （隐藏）层对输出误差的贡献程度。我们可以将其称为z2误差。
  - 通过应用我们的 S 型激活函数的导数来计算z2层的增量输出总和（就像步骤 2 ）
  - 通过对输入层与隐藏层 ( z2 ) 增量输出总和进行点积来调整第一层的权重。对于第二层，对隐藏层 ( z2 ) 与输出层 ( o ) 增量输出总和进行点积。
"""
