#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   
@File   :   test.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/20/22 4:54 PM   yinzikang      1.0         None
"""

import numpy as np
import matplotlib.pyplot as plt

# a = [1,2,3]
# b = [1,2,3]
# print([x*y for x in range(3) for y in range(3)])
# print([x*y for x,y in zip(a,b)])

# loc = [[0, 1], [2, 3]]
# scale = [[0, 0], [0, 0]]
# loc = [0, 1]
# scale = [0, 0]
# c = np.random.normal(loc, scale, size=(20, 2))
# print(c)
# plt.figure(1)
# plt.scatter(c[0, :], c[1, :])
# plt.show()
# c = np.array([[1, 2, 3], [4, 5, 6]])
# d = []
# d.append(c)
# d.append(c)
# d.append(c)
# print(d)
# e = np.concatenate(d,axis=1)
# print(e)

# c = np.array([[1, 2, 3], [4, 5, 6]])
# d = np.array([[1], [2]])
# e = np.linalg.norm(d-c,axis=0,keepdims=True)
# print(d-c)
# print(e)

mutation_operator_ratio = 0.5
generation_number = 200
adapted_ratio = np.zeros(generation_number)
for generation_index in range(generation_number):
    adapted_ratio[generation_index] = mutation_operator_ratio * 2 ** np.exp(
        1 - float(generation_number) / (generation_number + 1 - generation_index))

plt.figure(1)
plt.plot(adapted_ratio,label='mutation ratio init with 0.5')
plt.xlabel('generation index')
plt.ylabel('mutation ratio')
plt.title('mutation ratio vs generation index')
plt.legend()
plt.show()

# a = np.array([[1, 2], [1, 2], [1, 2]])
# b = np.array([[1, 2], [1, 2], [1, 2]]).transpose()
# print(a-b)
