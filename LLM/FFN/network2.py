import numpy as np
 
class FeedForwardNN:
  def __init__(self, input_size, hidden_size, output_size):
    # 初始化神经网络的参数
    self.weights1 = np.random.randn(input_size, hidden_size)
    self.bias1 = np.zeros((1, hidden_size))
    self.weights2 = np.random.randn(hidden_size, output_size)
    self.bias2 = np.zeros((1, output_size))
    
  def forward(self, x):
    # 前向传播，计算输出值
    self.z1 = np.dot(x, self.weights1) + self.bias1
    self.a1 = np.tanh(self.z1)
    self.z2 = np.dot(self.a1, self.weights2) + self.bias2
    self.output = self.z2
  
  def backward(self, x, y):
    # 反向传播，计算参数的梯度
    m = x.shape[0]
    delta2 = self.output - y
    dweights2 = np.dot(self.a1.T, delta2) / m
    dbias2 = np.sum(delta2, axis=0, keepdims=True) / m
    delta1 = np.dot(delta2, self.weights2.T) * (1 - np.power(self.a1, 2))
    dweights1 = np.dot(x.T, delta1) / m
    dbias1 = np.sum(delta1, axis=0) / m
      
    # 更新参数
    self.weights1 -= 0.1 * dweights1
    self.bias1 -= 0.1 * dbias1
    self.weights2 -= 0.1 * dweights2
    self.bias2 -= 0.1 * dbias2
  
  def train(self, x, y, epochs):
    # 训练神经网络
    for i in range(epochs):
      self.forward(x)
      self.backward(x, y)

  def predict(self, x):
    self.forward(x)
    return self.output