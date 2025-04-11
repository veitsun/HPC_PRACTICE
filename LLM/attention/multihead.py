import tensorflow as tf
import numpy as np

def MultiHeadAttention(x, M, P_q, P_k, P_v, P_o):
  q =  tf.einsum("d, hdk->hk", x, P_q) # 将输入向量 x 与查询映射张量 P_q 做爱因斯坦求和，生成h个头的查询表示
  K =  tf.einsum("md, hdk->hmk", M, P_k)
  V =  tf.einsum("md, hdv->hmv", M, P_v)
  logits = tf.einsum("hk,hmk->hm", q, K)
  weights = tf.nn.softmax(logits)
  o = tf.einsum("hm, hmv->hv", weights, V)
  y = tf.einsum("hv,hdv->d", o, P_o)
  return y



# 设置参数
d = 64    # 输入向量维度
m = 10    # 记忆矩阵长度
h = 8     # 注意力头数
k = 32    # 查询和键的维度
v = 32    # 值的维度

# 初始化测试数据
np.random.seed(42)

# 创建输入向量 (必须是一维的)
x = tf.constant(np.random.rand(d), dtype=tf.float32)

# 创建记忆矩阵
M = tf.constant(np.random.rand(m, d), dtype=tf.float32)

# 创建投影矩阵
P_q = tf.constant(np.random.rand(h, d, k), dtype=tf.float32)
P_k = tf.constant(np.random.rand(h, d, k), dtype=tf.float32)
P_v = tf.constant(np.random.rand(h, d, v), dtype=tf.float32)
P_o = tf.constant(np.random.rand(h, d, v), dtype=tf.float32)

# 打印每个张量的形状以进行验证
print("输入向量 x 形状:", x.shape)
print("记忆矩阵 M 形状:", M.shape)
print("查询投影 P_q 形状:", P_q.shape)
print("键投影 P_k 形状:", P_k.shape)
print("值投影 P_v 形状:", P_v.shape)
print("输出投影 P_o 形状:", P_o.shape)

# 调用多头注意力函数
result = MultiHeadAttention(x, M, P_q, P_k, P_v, P_o)

# 打印结果
print("\n计算结果形状:", result.shape)
print("前5个元素:", result[:5].numpy())

# 为了更好地理解，我们也可以手动检查一下中间步骤的结果形状
q = tf.einsum("d, hdk->hk", x, P_q)
K = tf.einsum("md, hdk->hmk", M, P_k)
V = tf.einsum("md, hdv->hmv", M, P_v)
logits = tf.einsum("hk,hmk->hm", q, K)
weights = tf.nn.softmax(logits)
o = tf.einsum("hm, hmv->hv", weights, V)

print("\n中间步骤形状验证:")
print("q 形状:", q.shape, "- 应为 [h, k]")
print("K 形状:", K.shape, "- 应为 [h, m, k]")
print("V 形状:", V.shape, "- 应为 [h, m, v]")
print("logits 形状:", logits.shape, "- 应为 [h, m]")
print("weights 形状:", weights.shape, "- 应为 [h, m]")
print("o 形状:", o.shape, "- 应为 [h, v]")
print("final 形状:", result.shape, "- 应为 [d]")