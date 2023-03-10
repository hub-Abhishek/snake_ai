import numpy as np
import pygame
import pdb
from scipy.spatial import distance

DIRECTIONS = ['left', 'right', 'up', 'down', ]
BLACK_BACKGROUND = (0, 0, 0)

class Game:
    def __init__(self,
                 food_count=1,
                 width=20, height=20, game_speed=30, display_option=True,
                  print_board=False, set_seed=True, seed=42) -> None:

        # game properties
        self.food_count = food_count
        self.width = width
        self.height = height

        self.snakes = []
        self.food_positions = []
        self.board = None
        self.player = []

        self.corners = np.array([[0, 0], [0, self.height-1], [self.width-1, 0], [self.width-1, self.height-1]])

        # random distinct position identifiers
        self.food_val = 3
        self.head_val = 2
        self.body_val = 1
        self.out_of_board = -100
        self.default_board_val = 0

        # display properties
        self.game_width = width * 20 + 40
        self.game_height = height * 20 + 40

        self.print_board_ = print_board
        self.game_speed = game_speed
        self.display_option = display_option

        if display_option:
            self.screen = pygame.display.set_mode((self.game_width, self.game_height))
            self.screen.fill(BLACK_BACKGROUND)
            self.food_image = pygame.image.load("img/food.png")
            pygame.display.set_caption('Snake')
            self.clock = pygame.time.Clock()

        # randomization aspects
        self.seed = seed
        self.set_seed = set_seed

        # lets go!!
        self.init_game()


    def init_game(self):
        if self.set_seed:
            np.random.seed(self.seed)
        self.food_positions = [self.get_empty_position()]
        self.update_board()

    def update_board(self):
        self.set_empty_board()
        if self.food_positions:
            for position in self.food_positions:
                self.board[tuple(position)] = self.food_val
        if len(self.snakes):
            for i, pos in enumerate(self.snakes):
                self.board[tuple(pos)] = self.head_val if i==0 else self.body_val

    def set_empty_board(self):
        self.board = np.zeros((self.width, self.height))

    def update_food(self):
        self.food_positions = [self.get_empty_position()]
        self.food_x = self.food_positions[0][0]
        self.food_y = self.food_positions[0][1]
        self.update_board()

    def get_empty_position(self):
        x_pos = np.random.randint(0, self.width)
        y_pos = np.random.randint(0, self.height)
        while not self.check_eligibility_of_position(x_pos, y_pos):
            x_pos = np.random.randint(0, self.width)
            y_pos = np.random.randint(0, self.height)
        return [x_pos, y_pos]

    def check_eligibility_of_position(self, x_pos, y_pos):
        if x_pos < self.width and y_pos < self.height:
            if not (any([[x_pos, y_pos] == z for z in self.food_positions])
                    or any([[x_pos, y_pos] == z for z in self.snakes])):
                return True
        else:
            return False

    def display_ui(self):
        myfont = pygame.font.SysFont('Segoe UI', 20)

        for i in range(len(self.player)):
            color = ""
            if self.player[i].color == "green":
                color = "Green"
            if self.player[i].color == "blue":
                color = "Blue"
            if self.player[i].color == "red":
                color = "Red"
            if self.player[i].color == "purple":
                color = "Purple"
            text_score = myfont.render(color + ' Snake Score: ' + str(self.player[i].score), True, (0, 0, 0))
            text_highest = myfont.render('Record: ' + str(self.player[i].record), True, (0, 0, 0))
            avg = self.player[i].total_score / (self.player[i].deaths + 1)
            text_avg = myfont.render('Avg: ' + str(round(avg)), True, (0, 0, 0))
            self.gameDisplay.blit(text_score, (35, 440 + i*20))
            self.gameDisplay.blit(text_highest, (230, 440 + i*20))
            self.gameDisplay.blit(text_avg, (340, 440 + i*20))

        self.gameDisplay.blit(self.bg, (10, 10))

    def display_food(self, x_pos, y_pos):
        self.screen.blit(self.food_image, (x_pos*20, y_pos*20))
        pygame.display.update()
        while 1==1:
            print(123)

    def display(self):
        if self.display_option:
            # self.display_ui()
            for player in self.player:
                player.display_player(self)
            for food in self.food_positions:
                self.display_food(*food)



if __name__=="__main__":
    game = Game()
    print(123)

#     def update_board(self):
#         for x_pos in range(self.width):
#             for y_pos in range(self.height):
#                 self.board[x_pos, y_pos] = self.food_val if
#
#
#     def set_food_positions(self):
#         for food in self.food_positions:
#             self.set_board_val(food, self.food_val)
#
#     def get_snakes(self, snake_heads, snake_lengths):
#         for i in range(len(snake_heads)):
#             head_position = snake_heads[i]
#             length = snake_lengths[i]
#             self.get_snake(head_position, length, i)
#
#     def set_board_val(self, pos, val):
#         if pos.shape==(2,):
#             self.board[pos[0], pos[1]] = val
#         else:
#             for p in pos:
#                 self.board[p[0], p[1]] = val
#
#
#     def get_snake(self, head_position, length, snake_num):
#         self.snakes[snake_num] = head_position.reshape(1, 2)
#         self.set_board_val(head_position, self.head_val)
#         next_eligible_root = head_position.copy()
#
#         for i in range(length-1):
#             next_eligible_cells = self.get_all_eligible_neighbors(next_eligible_root)
#             # print(f'recieved {next_eligible_cells}')
#             next_eligible_cells = self.get_cross(next_eligible_cells, next_eligible_root)
#             # print(f'recieved {next_eligible_cells}')
#             next_eligible_cells = self.remove_ineligible_neighbord(next_eligible_cells)
#             # print(f'recieved {next_eligible_cells}')
#             if not len(next_eligible_cells):
#                 a = 1
#                 # print(self.snakes)
#             else:
#                 next_eligible_root = next_eligible_cells[np.random.choice(np.arange(len(next_eligible_cells)))].squeeze()
#             # print(f'selectred {next_eligible_root}')
#             self.set_board_val(next_eligible_root, self.body_val)
#             self.snakes[snake_num] = np.append(self.snakes[snake_num], next_eligible_root.reshape(1,2), axis=0)
#
#
#     def get_all_eligible_neighbors(self, pos, borders=False, window_size=3):
#         if window_size%2 != 1:
#             window_size = window_size - 1
#
#         neighbors_h = [n for n in range(window_size)]
#         neighbors_h = np.array([n-neighbors_h[window_size//2] for n in neighbors_h])
#
#         neighbors_v = [n for n in range(window_size)]
#         neighbors_v = np.array([n-neighbors_v[window_size//2] for n in neighbors_v])
#
#         neighbors_h = neighbors_h + pos[0]
#         neighbors_v = neighbors_v + pos[1]
#
#         neighbors_h = self.position_mask(neighbors_h, borders).reshape([-1,  1])
#         neighbors_v = self.position_mask(neighbors_v, borders).reshape([ 1, -1])
#         # if borders:
#         #     import pdb; pdb.set_trace();
#
#         positions_h, positions_v = np.broadcast_arrays(neighbors_h, neighbors_v)
#         positions = np.stack((positions_h, positions_v), axis=-1).reshape(-1, 2)
#         positions = positions[(distance.cdist(positions, pos.reshape(1, 2), 'cityblock')!=0).reshape(-1)]
#
#         # deleted_positions = []
#         # for row, position in enumerate(positions):
#         #     # print(f'checking for {position} - {self.check_eligibility_of_position(position)}')
#         #     if self.check_eligibility_of_position(position):
#         #         deleted_positions.append(row)
#         # positions = np.delete(positions, deleted_positions, 0)
#
#         return positions
#
#     def get_cross(self, positions, pos):
#         return positions[(distance.cdist(positions, pos.reshape(1, 2), 'cityblock')==1).reshape(-1)]
#
#     def remove_ineligible_neighbord(self, positions):
#         deleted_positions = []
#         for row, position in enumerate(positions):
#             # print(f'checking for {position} - {self.check_eligibility_of_position(position)}')
#             if self.check_eligibility_of_position(position):
#                 deleted_positions.append(row)
#         positions = np.delete(positions, deleted_positions, 0)
#
#         return positions
#
#     def check_eligibility_of_position(self, position):
#         return self.extract_position(position)==self.food_val or self.extract_position(position)==self.head_val or self.extract_position(position)==self.body_val
#
#     def extract_position(self, position):
#         if position[0]>=0 and position[0]<self.size and position[1]>=0 and position[1]<self.size:
#             return self.board[position[0], position[1]]
#         else:
#             return self.out_of_board
#
#     def position_mask(self, arr, borders):
#         if not borders:
#             return arr[(arr>=0) & (arr<self.size)]
#         else:
#             return arr
#

#
#     def print_board(self, print_board_):
#         if print_board_ :
#             string = ''
#             # print(self.snakes)
#             for col in range(self.size-1, -1, -1):
#             # for col in range(self.size-1, -1, -1 ):
#                 row_str = ''
#                 for row in range(self.size):
#                     current_pos = np.array([row, col])
#                     if (self.food_positions==current_pos).all(axis=1).any():
#                         row_str += '|@ '
#                     elif (self.snake_heads==current_pos).all(axis=1).any():
#                         row_str += '|* '
#                     elif (self.snakes[0]==current_pos).all(axis=1).any():
#                         row_str += '|++'
#                     else:
#                         row_str += '|__'
#                 row_str += '|\n'
#                 string += row_str
#             print(string)
#
#
#     def move(self, dir=None, snake_num=0):
#         # pdb.set_trace()
#         if not self.zinda_hai_ki_nahi:
#             # print('died')
#             return
#         if dir is None:
#             dir = self.dir
#         if dir is None:
#             dir = np.random.choice(DIRECTIONS, 1)
#             self.dir = None
#         else:
#             self.dir = dir
#         self.moves_list.append(dir)
#         # print(f'Moving {dir}')
#
#         self.original_state = self.board.copy()
#         self.initial_food_positions = self.food_positions.copy()
#         self.initial_head_positions = self.snake_heads.copy()
#         self.initial_snake_positions = self.snakes.copy()
#
#         self.snake_heads[snake_num] = self.shift_head_in_dir(self.snake_heads[snake_num], dir)
#         # print(self.snake_heads[snake_num])
#         self.zinda_hai_ki_nahi = self.check_state(snake_num)
#
#         if not self.zinda_hai_ki_nahi:
#             return
#
#         if (self.snake_heads[snake_num]==self.initial_food_positions).all(axis=1).any():
#             # TODO:change food positions for multiple food positions
#             self.snakes[snake_num] = np.append(self.food_positions, self.snakes[snake_num], axis=0)
#             self.food_positions = self.get_positions(self.food_count)
#         else:
#             self.snakes[snake_num][1:] = self.snakes[snake_num][:-1]
#             self.snakes[snake_num][0] = self.snake_heads[snake_num]
#         self.update_board()
#
#
#     def check_state(self, snake_num):
#         # pdb.set_trace()
#         inside_board = ((self.snake_heads[snake_num]).min()<0) or ((self.snake_heads[snake_num]).max()>=self.size)
#         snake_bite = (self.snake_heads[snake_num]==self.snakes[snake_num][:-1]).all(axis=1).any()
#         # pdb.set_trace()
#         # if inside_board or snake_bite:
#             # print("you're dead")
#             # print(f"moves list: {self.moves_list}")
#             # print(f"snake size - {len(self.snakes[0])}")
#         return not (inside_board or snake_bite)
#
#     def shift_head_in_dir(self, snake_head, dir):
#         if dir == 'up':
#             new_head = snake_head + np.array([0, 1])
#         if dir == 'down':
#             new_head = snake_head - np.array([0, 1])
#         if dir == 'left':
#             new_head = snake_head - np.array([1, 0])
#         if dir == 'right':
#             new_head = snake_head + np.array([1, 0])
#
#         return new_head
#
#     def update_board(self):
#         self.board = np.zeros((self.size, self.size))
#         self.set_food_positions()
#         self.set_snake_positions()
#         self.print_board(self.print_board_)
#
#     def set_snake_positions(self):
#         for i, snake in enumerate(self.snakes):
#             self.set_board_val(snake, self.body_val)
#             self.set_board_val(self.snake_heads[i], self.head_val)
#
#
# if __name__=="__main__":
#     board_size, print_board = 10, True
#     game = Game(board_size, print_board=print_board, set_seed=False)
#     move_dict = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}
#     while game.zinda_hai_ki_nahi:
#         dir = input('dir:')
#         dir = move_dict.get(dir, None)
#         game.move(dir)