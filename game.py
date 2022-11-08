import numpy as np
import pdb
from scipy.spatial import distance
from scipy.ndimage import interpolation

class Game():
    def __init__(self, board_size=10, snake_count=1, food_count=1, snake_initial_size=3) -> None:
        self.size = board_size
        self.snake_count = snake_count
        self.snake_initial_size = snake_initial_size
        self.snake_lengths = [self.snake_initial_size] * self.snake_count
        self.food_count = food_count

        self.food_val = -10
        self.head_val = -5
        self.body_val = -4

        self.board = np.zeros((self.size, self.size))
        self.corners = np.array([[0, 0], [0, self.size-1], [self.size-1, 0], [self.size-1, self.size-1]])
        
        self.zinda_hai_ki_nahi = True
        self.snakes = [0] * self.snake_count
        
        self.get_initial_conditions()
        self.print_board()

    
    def get_initial_conditions(self, seed=42):
        np.random.seed(seed)

        self.initial_food_positions = self.get_positions(self.food_count)
        self.initial_head_positions = self.get_positions(self.snake_count) # np.array([[9, 6]]) # 
        
        self.food_positions = self.initial_food_positions.copy()
        self.set_food_positions()
        
        self.snake_heads = self.initial_head_positions.copy()
        self.get_snakes(self.snake_heads, self.snake_lengths, )
        self.initial_snake_positions = self.snakes.copy()

        self.dir = None

        print(f'food positions - {self.food_positions}')
        print(f'snake heads - {self.snake_heads}')
            
    def set_food_positions(self):
        for food in self.food_positions:
            self.set_board_val(food, self.food_val)

    def get_snakes(self, snake_heads, snake_lengths):
        for i in range(len(snake_heads)):
            head_position = snake_heads[i]
            length = snake_lengths[i]
            self.get_snake(head_position, length, i)

    def set_board_val(self, pos, val):
        if pos.shape==(2,):
            self.board[pos[0], pos[1]] = val
        else:
            for p in pos:
                self.board[p[0], p[1]] = val
            

    def get_snake(self, head_position, length, snake_num):
        self.snakes[snake_num] = head_position.reshape(1, 2)
        self.set_board_val(head_position, self.head_val)      
        next_eligible_root = head_position.copy()

        for i in range(length-1):            
            next_eligible_cells = self.get_all_eligible_neighbors(next_eligible_root)
            # print(f'recieved {next_eligible_cells}')
            next_eligible_root = next_eligible_cells[np.random.choice(np.arange(len(next_eligible_cells)))].squeeze()
            # print(f'selectred {next_eligible_root}')
            self.set_board_val(next_eligible_root, self.body_val)
            self.snakes[snake_num] = np.append(self.snakes[snake_num], next_eligible_root.reshape(1,2), axis=0)
        
        
    def get_all_eligible_neighbors(self, pos):
        neighbors_h = np.array([-1, 0, 1])
        neighbors_v = np.array([-1, 0, 1])

        neighbors_h = neighbors_h + pos[0]
        neighbors_v = neighbors_v + pos[1]

        neighbors_h = self.position_mask(neighbors_h).reshape([-1,  1])
        neighbors_v = self.position_mask(neighbors_v).reshape([ 1, -1])
        
        positions_h, positions_v = np.broadcast_arrays(neighbors_h, neighbors_v)        
        positions = np.stack((positions_h, positions_v), axis=-1).reshape(-1, 2)
        positions = positions[(distance.cdist(positions, pos.reshape(1, 2), 'cityblock')==1).reshape(-1)] 
        
        deleted_positions = []
        for row, position in enumerate(positions):
            # print(f'checking for {position} - {self.check_eligibility_of_position(position)}')
            if self.check_eligibility_of_position(position):
                deleted_positions.append(row)
        positions = np.delete(positions, deleted_positions, 0)

        return positions

    def check_eligibility_of_position(self, position):
        return self.extract_position(position)==self.food_val or self.extract_position(position)==self.head_val or self.extract_position(position)==self.body_val

    def extract_position(self, position):
        return self.board[position[0], position[1]]

    def position_mask(self, arr):
        return arr[(arr>=0) & (arr<self.size)]
    
    def get_positions(self, count, ):
        return np.random.randint(0, self.board.shape, (count, 2))

    def print_board(self):
        string = ''
        # print(self.snakes)
        for col in range(self.size-1, -1, -1):
        # for col in range(self.size-1, -1, -1 ):
            row_str = ''
            for row in range(self.size):
                current_pos = np.array([row, col])
                if (self.food_positions==current_pos).all(axis=1).any():
                    row_str += '|@ '
                elif (self.snake_heads==current_pos).all(axis=1).any():
                    row_str += '|* '
                elif (self.snakes[0]==current_pos).all(axis=1).any():
                    row_str += '|++'
                else:
                    row_str += '|__'
            row_str += '|\n'
            string += row_str
        print(string)


    def move(self, dir=None, snake_num=0):
        # pdb.set_trace()
        if not self.zinda_hai_ki_nahi:
            print('died')
            return
        if dir is None:
            dir = self.dir
        if dir is None:
            dir = np.random.choice(['up', 'down', 'left', 'right'], 1)
            self.dir = None
        print(f'Moving {dir}')

        self.original_state = self.board.copy()        
        self.initial_food_positions = self.food_positions.copy()
        self.initial_head_positions = self.snake_heads.copy()
        self.initial_snake_positions = self.snakes.copy()
        
        self.snake_heads[snake_num] = self.shift_head_in_dir(self.snake_heads[snake_num], dir)
        # print(self.snake_heads[snake_num])
        self.zinda_hai_ki_nahi = self.check_state(snake_num)
        
        if not self.zinda_hai_ki_nahi:
            return

        if (self.snake_heads[snake_num]==self.initial_food_positions).all(axis=1).any():
            # TODO:change food positions for multiple food positions
            self.snakes[snake_num] = np.append(self.food_positions, self.snakes[snake_num], axis=0) 
            self.food_positions = self.get_positions(self.food_count)
        else:
            self.snakes[snake_num][1:] = self.snakes[snake_num][:-1]
            self.snakes[snake_num][0] = self.snake_heads[snake_num]
        self.update_board()

    
    def check_state(self, snake_num):
        # pdb.set_trace()
        inside_board = ((self.snake_heads[snake_num]).min()<0) or ((self.snake_heads[snake_num]).max()>=self.size)
        snake_bite = (self.snake_heads[snake_num]==self.snakes[snake_num][:-1]).all(axis=1).any()
        # pdb.set_trace()
        if inside_board or snake_bite:
            print("you're dead")
        return not (inside_board or snake_bite)

    def shift_head_in_dir(self, snake_head, dir):
        if dir == 'up':
            new_head = snake_head + np.array([0, 1])
        if dir == 'down':
            new_head = snake_head - np.array([0, 1])
        if dir == 'left':
            new_head = snake_head - np.array([1, 0])
        if dir == 'right':
            new_head = snake_head + np.array([1, 0])
        
        return new_head
    
    def update_board(self):
        self.board = np.zeros((self.size, self.size))
        self.set_food_positions()
        self.set_snake_positions()
        self.print_board()

    def set_snake_positions(self):
        for i, snake in enumerate(self.snakes):
            self.set_board_val(snake, self.body_val)
            self.set_board_val(self.snake_heads[i], self.head_val)



board_size, snake_count, food_count = 10, 1, 1
game = Game(board_size, snake_count, food_count)
print(game.zinda_hai_ki_nahi)
# game.move('down')
# game.move('left')
# game.move()
# game.move()
# game.move()
# game.move()
# game.move('up')
# game.move()
# game.move()
# game.move('right')
# game.move()
# game.move()
# game.move()
# game.move()
# game.move('down')
# game.move()

# game.move('left')
# game.move('up')
# game.move()
while game.zinda_hai_ki_nahi:
    game.move()