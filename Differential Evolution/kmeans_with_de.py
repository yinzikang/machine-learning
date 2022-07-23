#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Target :   
@File   :   kmeans_with_de.py    
@author :   yinzikang

@Modify Time      @Author    @Version    @Description
------------      -------    --------    -----------
4/20/22 10:15 PM   yinzikang      1.0         None
"""
import numpy as np
import matplotlib.pyplot as plt
from differential_evolution import differential_evolution_algorithm as de
from improved_differential_evolution import differential_evolution_algorithm as ide


def data_generation(sample_number: dict = {}):
    # np.random.seed(16)
    return [np.random.normal(loc=sample_number['cluster_mean'][i],
                             scale=sample_number['cluster_std'][i],
                             size=(sample_number['cluster_number'][i], 2)).transpose()
            for i in range(len(sample_number['cluster_number']))]


def function(population, data):
    score = np.zeros(population.shape[0])
    # every individual
    for i in range(population.shape[0]):
        # every data
        for j in range(data.shape[1]):
            distance_to_centroids = np.linalg.norm(data[:, j].reshape(2, 1) - population[i], axis=0, keepdims=True)
            score[i] += distance_to_centroids.min()

    return score


dataset_information = dict(cluster_number=[15, 15, 15],
                           cluster_mean=[[-0.5, -0.4],
                                         [0.35, -0.5],
                                         [0.1, 0.35]],
                           cluster_std=[[0.2, 0.15],
                                        [0.15, 0.15],
                                        [0.2, 0.2]])
# dataset_information = dict(cluster_number=[1, 1, 1],
#                            cluster_mean=[[-0.5, -0.5],
#                                          [0.5, -0.5],
#                                          [0, 0.5]],
#                            cluster_std=[[0., 0.],
#                                         [0., 0.],
#                                         [0., 0.]])

data_raw = data_generation(dataset_information)
data_array = np.concatenate(data_raw, axis=1)

dimension = [2, 3]
limit_max = [[1.] * 3] * 2
limit_min = [[-1.] * 3] * 2

p_number = 50
g_number = 200
m_ratio = 0.3
c_ratio = 0.3

score_of_function_de, root_de = de(target_function=function,
                                   individual_dimension=dimension,
                                   individual_limitation=[limit_max, limit_min],
                                   population_number=p_number,
                                   generation_number=g_number,
                                   mutation_operator_ratio=m_ratio,
                                   crossover_operator_ratio=c_ratio,
                                   extra_data=data_array)

score_of_function_ide, root_ide = ide(target_function=function,
                                      individual_dimension=dimension,
                                      individual_limitation=[limit_max, limit_min],
                                      population_number=p_number,
                                      generation_number=g_number,
                                      mutation_operator_ratio=m_ratio,
                                      crossover_operator_ratio=c_ratio,
                                      extra_data=data_array)

print(score_of_function_de.min(), score_of_function_ide.min())
print(root_de, root_ide)

print(data_raw)
plt.figure(1)
for i in range(len(dataset_information['cluster_number'])):
    plt.scatter(data_raw[i][0, :], data_raw[i][1, :], label='class ' + str(i + 1))
plt.scatter(np.array(dataset_information['cluster_mean'])[:,0], np.array(dataset_information['cluster_mean'])[:,1],
            label='true value')
plt.scatter(root_de[0, :], root_de[1, :], label='de_prediction')
plt.scatter(root_ide[0, :], root_ide[1, :], label='ide_prediction')
plt.legend()

plt.figure(2)
plt.subplot(221)
plt.plot(score_of_function_de)
# plt.xlabel('generation')
plt.ylabel('de score')
plt.title('score of every individual')
plt.subplot(222)
plt.plot(score_of_function_de.min(axis=1))
# plt.xlabel('generation')
# plt.ylabel('score')
plt.title('best score of each generation')

# plt.figure(3)
plt.subplot(223)
plt.plot(score_of_function_ide)
plt.xlabel('generation')
plt.ylabel('score')
# plt.title('score of every individual')
plt.subplot(224)
plt.plot(score_of_function_ide.min(axis=1))
plt.xlabel('generation')
plt.ylabel('ide score')
# plt.title('best score of each generation')

plt.figure(4)
plt.plot(score_of_function_de.min(axis=1), label='score_of_function_de')
plt.plot(score_of_function_ide.min(axis=1), label='score_of_function_ide')
plt.title('de vs ide')
plt.legend()

plt.show()
