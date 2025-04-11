import tensorflow as tf

def DotProductAttention(q,K,V):
  """
  这段代码实现了点积注意力机制
  这是 transformer 模型和许多注意力机制的基础组件
  """
  logits = tf.einsum('qd,kd->qk', q, K) 
  weights = tf.nn.softmax(logits)
  return tf.einsum('qk,kv->qv', weights, V)



if __name__ == "__main__":
  # 测试代码
  print("------------------------------------------------------------------")
  q = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
  K = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.float32)
  V = tf.constant([[1, 2], [3, 4], [5, 6]], dtype=tf.float32)

  result = DotProductAttention(q,K,V)
  print("------------------------------------------------------------------")
  print(result)
  # 输出结果