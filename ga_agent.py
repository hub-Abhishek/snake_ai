import numpy as np
import math
import copy
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as f

from game import *
from player import *

PATH = 'C:/Users/abhis/Downloads/New folder/project/Snake AI'



class geneticPlayer:
    def __init__(self, pop_size, num_trials, mutation_rate, mutation_change,
                 input_window_size,
                 layers, output_size=4, num_gen=5, move_limit=100,
                 training=False):

        self.pop_size = pop_size  # population size
        self.num_trials = num_trials
        self.mutation_rate = mutation_rate
        self.mutation_change = mutation_change
        self.move_limit = move_limit
        self.num_gen = num_gen

        self.input_window_size = input_window_size
        self.input_size = (input_window_size * input_window_size) - 1 + 4
        # self.num_hidden_layers = num_hidden_layers
        self.layers = layers
        self.output_size = output_size
        self.training = training

        self.generate_brains(self.pop_size, self.layers)

    def mutate(self):
        pass

    def crossover(self):
        pass

def weights_size(units):
    s = 0
    for i in range(len(units)-1):
        s += units[i] * units[i+1]
    return s

def generate_brain(weights, layers):
    current_weights = weights.copy()
    model = nn.Sequential()
    for i, layer in enumerate(layers[:-1]):
        perceptrons = nn.Linear(layers[i], layers[i + 1])
        with torch.no_grad():
            perceptrons.weight = nn.Parameter(
                torch.Tensor(current_weights[:layers[i] * layers[i + 1]].reshape(perceptrons.weight.shape)))
        current_weights = current_weights[layers[i] * layers[i + 1]:]
        model.add_module(f'layer_{i}', perceptrons)
        model.add_module(f'activation_layer_{i}', nn.ReLU() if i != len(layers) - 2 else nn.Softmax())
    for layer in model.parameters():
        layer.requires_grad = False
    return model

def get_box(board, pos):
    x_pos = pos[0]
    y_pos = pos[1]


def get_state(player):
    player.init_game.update_board()
    board_current_state = player.init_game.board.copy()
    board_current_state = np.pad(board_current_state, 1, 'constant', constant_values=1)

    x_pos = player.init_game.snakes[0][0]
    y_pos = player.init_game.snakes[0][1]
    box = board_current_state[x_pos:x_pos + 3, y_pos:y_pos + 3].flatten()
    box = box==1
    state = np.hstack([box[:4],
                       box[5:],
                       player.direction,
                       player.where_food
                       ])
    return torch.tensor(state).float()


def run_for_one_individual(weights, layers, max_moves, max_steps_per_food):
    game, player = initialize_game_with_player(False)
    brain = generate_brain(weights, layers)
    food_score = 0
    total_food_score = 0
    max_food_score = 0
    current_move = 0
    deaths = 0
    slow_penalty = 0

    for i in range(max_moves):
        if player.verbose_level >= 3:
            print(f'snake - ', player.init_game.snakes)
            print(f'food - ', player.init_game.food_positions)
        state = get_state(player)
        direction_scores = brain(state)
        direction = direction_scores.argmax()
        player.move(DIRECTIONS[direction])
        if player.verbose_level>=3:
            print(DIRECTIONS[direction])
        if not player.aa_aa_aa_aa_stayin_alive:
            player = restart_for_game(game)
            deaths += 1
            continue
        if player.ate_in_last_move == 1:
            current_move = 0
            total_food_score += 1
        else:
            current_move += 1

        if current_move >= max_steps_per_food:
            player = restart_for_game(game)
            slow_penalty += 1

        food_score = len(player.init_game.snakes) - 3
        if food_score > max_food_score:
            max_food_score = food_score

    return deaths * (-150) + max_food_score * 5000 + slow_penalty * (-1000) + int(max_moves / (total_food_score + 1)) * (-100), \
           deaths, total_food_score / (deaths + 1), max_food_score
def run_for_one_generation(pop_size, all_weights, layers, move_limit, max_steps_per_food, generation):
    fitness = []
    deaths = []
    avg_score = []
    max_scores = []
    for individual in range(pop_size):
        weights = all_weights[individual]
        fit, snake_deaths, snake_avg_score, record = run_for_one_individual(weights, layers, move_limit, max_steps_per_food)
        snake_avg_score = round(snake_avg_score, 2)
        print('generation: ' + str(generation) + ' fitness value of snake ' + str(individual) + ':  ' + str(fit) +
              '   Deaths: ' + str(snake_deaths) + '   Avg score: ' + str(snake_avg_score) + '   Record: ' + str(record))
        fitness.append(fit)
        deaths.append(snake_deaths)
        avg_score.append(snake_avg_score)
        max_scores.append(record)
    return np.array(fitness), np.array(deaths), np.array(avg_score), np.array(max_scores)


# def best_brains(all_brains, fitness, top_n):
#     temp_fitness = np.array(fitness, copy=True)
#     parents = np.empty((top_n, all_brains.shape[1]))
#     for parent_num in range(top_n):
#         max_fitness_idx = np.where(temp_fitness == np.max(temp_fitness))
#         max_fitness_idx = max_fitness_idx[0][0]
#         parents[parent_num, :] = all_brains[max_fitness_idx, :]
#         temp_fitness[max_fitness_idx] = -99999999
#     return parents

def select_mating_pool(pop, fitness, num_parents):
    temp_fitness = np.array(fitness, copy=True)
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(temp_fitness == np.max(temp_fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        temp_fitness[max_fitness_idx] = -99999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    for k in range(offspring_size[0]):
        while True:
            parent1_idx = random.randint(0, parents.shape[0] - 1)
            parent2_idx = random.randint(0, parents.shape[0] - 1)
            # produce offspring from two parents if they are different
            if parent1_idx != parent2_idx:
                for j in range(offspring_size[1]):
                    if random.uniform(0, 1) < 0.5:
                        offspring[k, j] = parents[parent1_idx, j]
                    else:
                        offspring[k, j] = parents[parent2_idx, j]
                break
    return offspring

# mutating the offsprings generated from crossover to maintain variation in the population
def mutation(offspring_crossover, mutations=1):
    for idx in range(offspring_crossover.shape[0]):
        for _ in range(mutations):
            i = random.randint(0, offspring_crossover.shape[1]-1)
            random_value = np.random.choice(np.arange(-1, 1, step=0.001), size=1, replace=False)
            offspring_crossover[idx, i] = offspring_crossover[idx, i] + random_value
    return offspring_crossover


if __name__=="__main__":
    num_gen = 100
    pop_size = 50
    mutation_rate = 0.1
    mutation_change = 0.1
    mutations = 1 # TODO: replace
    population_name = 'test'
    layers = [16, 120, 120, 120, 4]
    max_steps_per_food = 200
    move_limit = 2000
    top_n = int(pop_size/4)
    training = True

    if training:
        num_weights = weights_size(layers)
        all_param_shape = (pop_size, num_weights)
        all_initial_brains = np.random.choice(np.arange(-1, 1, 0.01), size=all_param_shape)
        all_brains = all_initial_brains.copy()

        # for one generation for one individual
        for generation in range(num_gen):
            fitness, deaths, avg_score, max_scores = run_for_one_generation(pop_size, all_brains, layers, move_limit, max_steps_per_food, generation)

            # print generation stats
            print('fittest snake in geneneration ' + str(generation) + ' : ', np.max(fitness))
            print('highest average score in geneneration ' + str(generation) + ' : ', np.max(avg_score))
            print('average fitness value in geneneration ' + str(generation) + ' : ', np.sum(fitness) / pop_size)
            print('average deaths in geneneration ' + str(generation) + ' : ', np.sum(deaths) / pop_size)
            print('average score in geneneration ' + str(generation) + ' : ', np.sum(avg_score) / pop_size)
            print('max score in geneneration ' + str(generation) + ' : ', max_scores[np.argmax(max_scores)])

            ranks = fitness.argsort()

            # survivors = all_brains[ranks][:top_n]
            survivors = select_mating_pool(all_brains, fitness, top_n)
            offspring_crossover = crossover(survivors, offspring_size=(all_brains.shape[0] - survivors.shape[0], num_weights))
            # adding some variations to the offspring using mutation.
            offspring_mutation = mutation(offspring_crossover, mutations)
            # creating the new population based on the parents and offspring
            all_brains[0:survivors.shape[0], :] = survivors
            all_brains[survivors.shape[0]:, :] = offspring_mutation

            # save generation stats
            dir_path = "weights/genetic_algorithm/" + str(population_name) + "/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            path = "weights/genetic_algorithm/" + str(population_name) + "/stats.txt"
            f = open(path, "a+")
            f.write(str(generation) + "\n")
            f.write(
                str(np.max(fitness)) + " " + str(np.max(avg_score)) + " " + str(np.sum(fitness) / pop_size) + " " +
                str(np.sum(deaths) / pop_size) + " " + str(np.sum(avg_score) / pop_size) + " " +
                str(max_scores[np.argmax(max_scores)]) + " " + str(np.argmax(fitness)) + " \n")
            f.close()
    print(123)



#
#     def train_for_n_gen(self):
#         num_gen = self.num_gen
#         for gen in range(num_gen):
#             max_moves, max_scores, avg_scores = self.one_gen(board_size, snake_count, food_count, set_seed, gen,
#                                                              window_size=self.input_window_size)
#             # print(max_scores[np.argsort(max_scores)][::-1][:math.ceil(len(max_scores)/4)])
#             # top_25 = np.argsort(max_scores)[::-1][:math.ceil(len(max_scores)/4)]
#             top_25 = np.argsort(avg_scores)[::-1][:math.ceil(len(avg_scores) / 4)]
#             print(f"**END OF GEN {gen}**")
#             print(f'max scores at the end of gen {gen} - {avg_scores}')
#             print(f'max moves at the end of gen {gen} - {max_moves}')
#             self.evolve(top_25)
#
#     def evolve(self, top_25):
#         new_brains = []
#         for brain in top_25:
#             new_brains.append(copy.deepcopy(self.brains[brain]))
#             new_brains.append(self.mutate(self.brains[brain]))
#         self.brains = new_brains
#         remaining = self.pop_size - len(self.brains)
#         self.brains += self.generate_brains(remaining, self.input_size, self.num_hidden_layers, self.hidden_layers_size,
#                                             self.output_size)
#
#     def mutate(self, brain):
#         for layer in brain:
#             if isinstance(layer, nn.Linear):
#                 # new_weights = layer.weight + ((torch.rand(layer.weight.shape)<self.mutation_rate).long() * torch.normal(layer.weight.shape) * self.mutation_change)
#                 new_weights = layer.weight + (
#                             (torch.rand(layer.weight.shape) < self.mutation_rate).long() * torch.empty(
#                         layer.weight.shape).normal_(mean=0, std=1) * self.mutation_change)
#                 layer.weight = nn.Parameter(new_weights, requires_grad=False)
#         return brain
#         # brain[1].weight = nn.Parameter(torch.rand(10, 12))
#         # (torch.rand(10, 12)>0.5).long()
#         # pass
#
#     def save_brains(self, PATH):
#         for i, brain in enumerate(self.brains):
#             torch.save(brain, f'{PATH}_brain_{i}.pt')
#
#     def load_brains(self, PATH):
#         for pop in range(self.pop_size):
#             self.brains[pop] = torch.load(f'{PATH}_brain_{pop}.pt')
#             self.brains[pop].eval()
#
#
# if __name__ == '__main__':
#     pop_size = 500
#     num_trials = 100
#     move_limit = 100
#     num_gen = 500
#
#     mutation_rate = 0.15
#     mutation_change = 0.1
#     input_window_size = 5
#     # input_size = input_window_size + 4 # input window size + food location
#
#     num_hidden_layers = 4
#     hidden_layers_size = [25] * num_hidden_layers
#     output_size = 4
#
#     board_size, snake_count, food_count, set_seed = 10, 1, 1, False
#     # game = Game(board_size, snake_count, food_count)
#
#     player = geneticPlayer(pop_size, num_trials, mutation_rate, mutation_change, input_window_size,
#                            num_hidden_layers, hidden_layers_size, output_size, num_gen, move_limit)
#
#     # player.one_gen(board_size, snake_count, food_count, set_seed)
#     player.train_for_n_gen()
#     player.save_brains(f'{PATH}/model')
#     # player.generate_input(game)
#     # print(player.brains[-1])
#     # print(torch.tensor(np.random.rand(1, 12)).float().shape)
#     # print(player.brains[-1](torch.tensor(np.random.rand(1, 12)).float()).argmax())