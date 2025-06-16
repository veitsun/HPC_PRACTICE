import sys
sys.path.append('../build')  # 添加模块所在路径

import my_module

# 使用函数
result = my_module.add(5, 3)
print(f"5 + 3 = {result}")

# product = my_module.multiply(2.5, 4.0)
# print(f"2.5 * 4.0 = {product}")

# # 使用列表处理函数
# input_list = [1, 2, 3, 4, 5]
# output_list = my_module.process_list(input_list)
# print(f"Processed list: {output_list}")

# # 使用类
# calc = my_module.Calculator(10.0)
# print(f"Initial value: {calc.value}")
# print(f"10 + 5 = {calc.add(5)}")
# print(f"10 * 3 = {calc.multiply(3)}")