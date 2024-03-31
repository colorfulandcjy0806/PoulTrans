def reverse_list(lst):
    return lst[::-1]

# 测试函数
lst = [1, 2, 3, 4, 5]
print(reverse_list(lst))  # 输出 [5, 4, 3, 2, 1]

def reverse_list(lst):
    return list(reversed(lst))

# 测试函数
lst = [1, 2, 3, 4, 5]
print(reverse_list(lst))  # 输出 [5, 4, 3, 2, 1]
