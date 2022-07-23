#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   
@File   :   trace_plan.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/18/22 3:22 PM   yinzikang      1.0         None
"""

import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution_algorithm as de
from improved_differential_evolution import differential_evolution_algorithm as ide


def function(population, data=None):
    """
    轨迹规划中的
    :param data:
    :param population:
    :return:
    """
    # score = np.array([np.linalg.norm(population[i], ord=np.inf) for i in range(population.shape[0])])
    score = np.zeros(population.shape[0])
    for i in range(population.shape[0]):
        score[i] = 1.1 * np.abs(np.diff(population[i])).sum()
        # reward
        if -1.5 < population[i, 0, 2] < -0.5 and -1.5 < population[i, 1, 2] < -0.5:
            score[i] -= 5
        if 1 < population[i, 0, 6] < 2 and 1 < population[i, 1, 6] < 2:
            score[i] -= 20
        # punishment
        if 0.5 < population[i, 0, 4] < 1.5 and 0.5 < population[i, 1, 4] < 1.5:
            score[i] += 50
        if -1 < population[i, 0, 8] < 1 and -0.5 < population[i, 1, 8] < 0.5:
            score[i] += 2

    return score


def plot_box(axis, xmin, xmax, zmin, zmax, y, color):
    box_x = np.array([xmin, xmax, xmax, xmin, xmin])
    box_y = np.ones(5) * y + 1
    box_z = np.array([zmin, zmin, zmax, zmax, zmin])
    axis.plot(box_x, box_y, box_z, color=color)


def plot_wall(axis, xmin, xmax, zmin, zmax, y, color):
    wall_x = np.array([xmin, xmax, xmax, xmin])
    wall_z = np.array([zmin, zmin, zmax, zmax])
    wall_y = wall_x * wall_z * 0 + y + 1
    axis.plot_surface(wall_x, wall_y, wall_z, color=color)


dimension = [2, 10]  # x,z * t
limit_max = [[0., 20., 20., 20., 20., 20., 20., 20., 20., 0.25],
             [0., 20., 20., 20., 20., 20., 20., 20., 20., 0.25]]
limit_min = [[0., -20., -20., -20., -20., -20., -20., -20., -20., -0.25],
             [0., -20., -20., -20., -20., -20., -20., -20., -20., -0.25]]
p_number = 150
g_number = 400
m_ratio = 0.25
c_ratio = 0.1
np.random.seed(32)
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
print(root_de, root_ide)

plt.figure(1)
plt.subplot(221)
plt.plot(score_of_function_de)
# plt.xlabel('generation')
plt.ylabel('de score')
plt.title('score of every individual')
plt.subplot(222)
plt.plot(score_of_function_de.min(axis=1))
# plt.xlabel('generation')
# plt.ylabel('de score')
plt.title('best score of each generation')


plt.subplot(223)
plt.plot(score_of_function_ide)
plt.xlabel('generation')
plt.ylabel('ide score')
# plt.title('score of every individual')
plt.subplot(224)
plt.plot(score_of_function_ide.min(axis=1))
plt.xlabel('generation')
# plt.ylabel('ide score')
# plt.title('best score of each generation')

plt.figure(3)
plt.plot(score_of_function_de.min(axis=1), label='score_of_function_de')
plt.plot(score_of_function_ide.min(axis=1), label='score_of_function_ide')
plt.title('de vs ide')
plt.legend()

plt.figure(4)
ax = plt.axes(projection='3d')

ax.plot(root_de[0], np.linspace(1, 10, 10), root_de[1], label='trace_ed')
ax.plot(root_ide[0], np.linspace(1, 10, 10), root_ide[1], label='trace_ied')
plt.legend()

plot_box(ax, -0.01, 0.01, -0.01, 0.01, 0, 'yellow')
plot_box(ax, -0.25, 0.25, -0.25, 0.25, 9, 'yellow')
plot_box(ax, -1.5, -0.5, -1.5, -0.5, 2, 'green')
plot_box(ax, 1, 2, 1, 2, 6, 'green')
plot_box(ax, 0.5, 1.5, 0.5, 1.5, 4, 'red')
plot_box(ax, -1, 1, -0.5, 0.5, 8, 'red')

ax.set_xlim(-3, 3)
ax.set_ylim(0, 10)
ax.set_zlim(-3, 3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
