# import numpy as np
# import math
# import copy
# import torch
# import torch.nn as nn
# import torch.nn.functional as f
#
from game import *
import pygame
import numpy as np

PATH = 'C:/Users/abhis/Downloads/New folder/project/Snake AI/'


class Player:
    def __init__(self, game, food_count=1,
                 snake_initial_size=3, max_turns=100,
                 verbose_level=2, display_option=True,
                 print_board=False,
                 set_seed=True, seed=42,):

        self.init_game = game
        self.food_count = food_count

        self.snake_initial_size = snake_initial_size
        self.max_turns = max_turns
        # self.zinda_hai_ki_nahi = True
        self.aa_aa_aa_aa_stayin_alive = True

        self.verbose_level = verbose_level
        self.seed = seed
        self.set_seed = set_seed
        self.display_option = display_option
        self.print_board = print_board

        self.color = 'yellow'
        self.snake_face_image = pygame.image.load('img/face.png')
        self.snake_face_image.set_colorkey((255, 255, 255))
        self.snake_face_image.set_alpha(128)

    def init_player(self, game=None):
        if game != None:
            self.init_game = game

        head = self.init_game.get_empty_position()
        self.init_game.snakes.append(head)
        for _ in range(self.snake_initial_size - 1):
            body_part = self.get_snake_body(self.init_game.snakes[-1])
            self.init_game.snakes.append(body_part)
        self.check_direction()
        print(123)

    def get_snake_body(self, current_pos):
        cross = [[1,0], [0,1], [-1,0], [0,-1]]
        all_options = [[x + y for x, y in zip(current_pos, body)] for body in cross]
        possible_options = [option if self.init_game.check_eligibility_of_position(*option) else '' for option in all_options]
        next = possible_options[np.random.choice(range(len(possible_options)))]
        return next

    def move(self, dir=None):
        if not self.aa_aa_aa_aa_stayin_alive:
            if self.verbose_level >= 2:
                print('You\'re already dead mister!')
            return self.aa_aa_aa_aa_stayin_alive
        if dir==None:
            dir = self.last_move
        original_snakes = self.init_game.snakes.copy()
        current_head = self.init_game.snakes[0]
        move_dir = {
            'left':  [-1,  0],
            'right': [ 1,  0],
            'up':    [ 0, -1],
            'down':  [ 0,  1],
        }
        new_head = [x + y for x, y in zip(current_head, move_dir[dir])]
        self.init_game.snakes = [new_head] + self.init_game.snakes
        self.aa_aa_aa_aa_stayin_alive = self.check_eligibility_of_new_head(*new_head)
        if not self.aa_aa_aa_aa_stayin_alive:
            self.init_game.snakes = original_snakes
            if self.verbose_level >= 2:
                print('You died mister!')
            return self.aa_aa_aa_aa_stayin_alive
        if any([new_head == z for z in self.init_game.food_positions]):
            self.init_game.update_food()
        else:
            self.init_game.snakes.pop(-1)
        self.last_move = dir
        self.check_direction()
        return self.aa_aa_aa_aa_stayin_alive

    def check_eligibility_of_new_head(self, x_pos, y_pos):
        if x_pos >=0 and x_pos < self.init_game.width and y_pos >=0 and y_pos < self.init_game.height:
            pass
        else:
            False
        if any([[x_pos, y_pos] == z for z in self.init_game.snakes[1:]]):
                return False
        else:
            return True

    def check_direction(self):
        directions = {
            'left':  [ 1,  0],
            'right': [-1,  0],
            'up':    [ 0, -1],
            'down':  [ 0,  1],
        }
        encoder = {
            'left':  [1, 0, 0, 0],
            'right': [0, 1, 0, 0],
            'up':    [0, 0, 1, 0],
            'down':  [0, 0, 0, 1],
        }
        direction = [y - x for x, y in zip(self.init_game.snakes[0], self.init_game.snakes[1])]
        for k, v in directions.items():
            if v == direction:
                self.direction = encoder[k]

        self.where_food = [self.init_game.food_positions[0][0] < self.init_game.snakes[0][0],
                           self.init_game.food_positions[0][0] > self.init_game.snakes[0][0],
                           self.init_game.food_positions[0][1] < self.init_game.snakes[0][0],
                           self.init_game.food_positions[0][0] > self.init_game.snakes[0][0]]

    def display_player(self, ):

        self.init_game.screen.blit(self.snake_face_image, tuple([pos*20 for pos in self.init_game.snakes[0]]))
        self.init_game.screen
        pygame.display.flip()
        self.init_game.clock.tick(10)


def initialize_game_with_player():
    game = Game()
    player = Player(game=game)
    player.init_player()
    player.init_game.player.append(player)
    return game, player


if __name__=="__main__":
    game, player = initialize_game_with_player()
    player = Player(game=game)
    player.init_player()
    player.init_game.player.append(player)
    player.move('up')
    # player.init_game.display()
    # print(123)

#     def generate_brains(self, pop_size, input_size, num_hidden_layers, hidden_layers_size, output_size):
#         brains = []
#         for brain in range(pop_size):
#             # AI_bc
#             AI_bc = self.generate_individual_brain(input_size, num_hidden_layers, hidden_layers_size, output_size)
#             brains.append(AI_bc)
#         return brains
#
#     def generate_individual_brain(self, input_size, num_hidden_layers, hidden_layers_size, output_size):
#         layers = []
#         layers.append(nn.LayerNorm(input_size))
#         layers.append(nn.Linear(input_size, hidden_layers_size[0]))
#         for layer in range(num_hidden_layers-1):
#             layers.append(nn.Linear(hidden_layers_size[layer], hidden_layers_size[layer+1]))
#             layers.append(nn.Tanh())
#         layers.append(nn.Linear(hidden_layers_size[-1], output_size))
#         # layers.append(nn.Tanh())
#         layers.append(nn.Softmax())
#         AI_bc = nn.Sequential(*layers, )
#         for layer in AI_bc.parameters():
#             layer.requires_grad = False
#         return AI_bc
#
#     def generate_input(self, game, window_size):
#
#         head = game.snake_heads[0]
#         positions = game.get_all_eligible_neighbors(head, borders=True, window_size=window_size)
#         input_vector = []
#         for i, position in enumerate(positions):
#             # import pdb; pdb.set_trace();
#             input_vector_val = game.extract_position(position)
#             if input_vector_val == game.food_val:
#                 input_vector_val = 1
#             elif input_vector_val == game.body_val:
#                 input_vector_val = -1
#             elif input_vector_val == game.default_board_val:
#                 input_vector_val = 0
#             elif input_vector_val == game.out_of_board:
#                 input_vector_val = -1
#             input_vector.append(input_vector_val)
#         input_vector.append(head[0])
#         input_vector.append(head[1])
#         input_vector.append(game.food_positions[0][0])
#         input_vector.append(game.food_positions[0][1])
#         input_vector = torch.tensor(input_vector).float().reshape(-1, self.input_size)
#         # print(input_vector.shape)
#         return input_vector
#
#     def get_move(self, input, i):
#         dir = self.brains[i](input).argmax()
#         return DIRECTIONS[dir]
#         # try:
#         #     dir = self.brains[i](input).argmax()
#         #     return DIRECTIONS[dir]
#         # except:
#         #     a = 1
#
#     def run_one_trial(self, board_size, snake_count, food_count, set_seed, i, window_size):
#         moves = 0
#         game = Game(board_size, snake_count, food_count, set_seed=set_seed)
#         while game.zinda_hai_ki_nahi and moves <= self.move_limit:
#             input = self.generate_input(game, window_size)
#             # print(f'snake head: {game.snake_heads[0]}')
#             # print(f'snake: {game.snakes}')
#             # print(f'input: {input}')
#             # print(f'matmul: {self.brains[i](input)}')
#             dir = self.get_move(input, i)
#             game.move(dir)
#             moves += 1
#         return moves, len(game.snakes[0])
#
#     def one_gen(self, board_size, snake_count, food_count, set_seed, gen=0, window_size=3):
#         max_moves = [0] * self.pop_size
#         max_scores = [0] * self.pop_size
#         avg_scores = [[]] * self.pop_size
#         for i in range(self.pop_size):
#             for j in range(self.num_trials):
#                 # print(f'trial {j} for pop {i} for gen {gen}')
#                 moves, score = self.run_one_trial(board_size, snake_count, food_count, set_seed, i, window_size)
#                 avg_scores[i].append(score)
#                 if moves > max_moves[i]:
#                     max_moves[i] = moves
#                 if score > max_scores[i]:
#                     max_scores[i] = score
#             avg_scores[i] = sum(avg_scores[i])/len(avg_scores[i])
#         # print(f'max moves for gen {gen} - {max_moves}')
#         # print(f'max scores  for gen {gen} - {max_scores}')
#         return np.array(max_moves), np.array(max_scores), np.array(avg_scores)
#
#
#     def train_for_n_gen(self):
#         num_gen = self.num_gen
#         for gen in range(num_gen):
#             max_moves, max_scores, avg_scores = self.one_gen(board_size, snake_count, food_count, set_seed, gen, window_size=self.input_window_size)
#             # print(max_scores[np.argsort(max_scores)][::-1][:math.ceil(len(max_scores)/4)])
#             # top_25 = np.argsort(max_scores)[::-1][:math.ceil(len(max_scores)/4)]
#             top_25 = np.argsort(avg_scores)[::-1][:math.ceil(len(avg_scores)/4)]
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
#         self.brains += self.generate_brains(remaining,  self.input_size, self.num_hidden_layers, self.hidden_layers_size, self.output_size)
#
#
#     def mutate(self, brain):
#         for layer in brain:
#             if isinstance(layer, nn.Linear):
#                 # new_weights = layer.weight + ((torch.rand(layer.weight.shape)<self.mutation_rate).long() * torch.normal(layer.weight.shape) * self.mutation_change)
#                 new_weights = layer.weight + ((torch.rand(layer.weight.shape)<self.mutation_rate).long() * torch.empty(layer.weight.shape).normal_(mean=0,std=1) * self.mutation_change)
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
# if __name__=='__main__':
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
#     hidden_layers_size = [25] *  num_hidden_layers
#     output_size = 4
#
#     board_size, snake_count, food_count, set_seed = 10, 1, 1, False
#     # game = Game(board_size, snake_count, food_count)
#
#     player = geneticPlayer(pop_size, num_trials, mutation_rate, mutation_change, input_window_size,
#                  num_hidden_layers, hidden_layers_size, output_size, num_gen, move_limit)
#
#     # player.one_gen(board_size, snake_count, food_count, set_seed)
#     player.train_for_n_gen()
#     player.save_brains(f'{PATH}/model')
#     # player.generate_input(game)
#     # print(player.brains[-1])
#     # print(torch.tensor(np.random.rand(1, 12)).float().shape)
#     # print(player.brains[-1](torch.tensor(np.random.rand(1, 12)).float()).argmax())