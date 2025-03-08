import numpy as np
from network import neural_network

# import our training data

X_all = np.array(([2, 9], [1, 5], [3, 6], [5, 10]), dtype=float) # 输入数据
y = np.array(([92], [86], [89]), dtype=float) # 输出数据

# 我们需要缩放数据，以确保所有数据点都在 0-1之间。这样的特征规范化是训练机器学习模型时时预处理的重要部分

X_all = X_all/np.max(X_all, axis=0) # 将 X_all 缩放到 0-1 之间
y = y/100 # 将 y 缩放到 0-1 之间

# 将数据分为训练集和测试集
X = np.split(X_all, [3])[0] # 训练集
X_test = np.split(X_all, [3])[1] # 测试集

print(X)
print(X_test)

nn = neural_network()
# o = nn.forward(X)
for i in range(150000): # 训练 1000 次
  print("Input: \n" + str(X))
  print("Actual Output: \n" + str(y))
  print("Predicted Output: \n" + str(nn.forward(X)))
  print("Loss: \n" + str(np.mean(np.square(y - nn.forward(X))))) # 取 square 的原因是 有的误差是负数，这个损失值只是一种量化我们与“完美”神经网络之间距离的方法。
  print("\n")
  nn.train(X, y)

nn.saveWeights()
nn.predictfunction(X_test)
# print("Predicted Output: \n" + str(o))
# print("Actual Output: \n" + str(y))

