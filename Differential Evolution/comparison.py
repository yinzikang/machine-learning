#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   
@File   :   comparison.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/20/22 3:43 PM   yinzikang      1.0         None
"""

import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution_algorithm as de
from improved_differential_evolution import differential_evolution_algorithm as ide


def function(population,data=None):
    """
    轨迹规划中的
    :param population:
    :return:
    """
    # score = np.array([np.linalg.norm(population[i], ord=np.inf) for i in range(population.shape[0])])

    score = [3 * np.cos(population[i, 0, 0] * population[i, 1, 0]) + population[i, 0, 0] + population[i, 1, 0]
             for i in range(population.shape[0])]
    # score = [- np.abs(np.sin(population[i, 0, 0]) *
    #                   np.cos(population[i, 1, 0]) *
    #                   np.exp(np.abs(1 - np.linalg.norm(population[i]) / np.pi)))
    #          for i in range(population.shape[0])]
    # BUKIN FUNCTION N. 6 x1 ∈ [-15, -5], x2 ∈ [-3, 3].
    # score = [100 * np.sqrt(np.abs(population[i, 1, 0] - 0.01 * population[i, 0, 0] ** 2))
    #          + 0.01 * np.abs(population[i, 0, 0] + 10)
    #          for i in range(population.shape[0])]
    # HOLDER TABLE FUNCTION  xi ∈ [-10, 10], for all i = 1, 2.
    # score = [- np.abs(np.sin(population[i, 0, 0]) *
    #                   np.cos(population[i, 1, 0]) *
    #                   np.exp(np.abs(1 - np.linalg.norm(population[i]) / np.pi)))
    #          for i in range(population.shape[0])]
    return score


dimension = [2, 1]
limit_max = [[4.], [4.]]
limit_min = [[-4.], [-4.]]
# limit_max = [[10.], [10.]]
# limit_min = [[-10.], [-10.]]
# limit_max = [[5.], [3.]]
# limit_min = [[-15.], [-3.]]

p_number = 30
g_number = 30
m_ratio = 0.3
c_ratio = 0.1

score_of_function_de, root_de = de(target_function=function,
                                   individual_dimension=dimension,
                                   individual_limitation=[limit_max, limit_min],
                                   population_number=p_number,
                                   generation_number=g_number,
                                   mutation_operator_ratio=m_ratio,
                                   crossover_operator_ratio=c_ratio)

score_of_function_ide, root_ide = ide(target_function=function,
                                      individual_dimension=dimension,
                                      individual_limitation=[limit_max, limit_min],
                                      population_number=p_number,
                                      generation_number=g_number,
                                      mutation_operator_ratio=m_ratio,
                                      crossover_operator_ratio=c_ratio)

print(score_of_function_de.min(), score_of_function_ide.min())
# print(root_de, root_ide)

plt.figure(3)
plt.plot(score_of_function_de.min(axis=1), label='score_of_function_de')
plt.plot(score_of_function_ide.min(axis=1), label='score_of_function_ide')
plt.title('de vs ide')
plt.legend()

plt.show()
