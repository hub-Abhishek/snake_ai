import numpy as np


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
        self.snakes = [[[]]] * self.snake_count
        
        self.get_initial_conditions()
        # print(self.board)
        self.print_board()

    
    def get_initial_conditions(self, seed=42):
        np.random.seed(seed)

        self.initial_food_positions = self.get_positions(self.food_count)
        self.initial_head_positions = self.get_positions(self.snake_count) # np.array([[0, 8]]) # 
        
        self.food_positions = self.initial_food_positions.copy()
        self.set_food_positions()
        
        self.snake_heads = self.initial_head_positions.copy()
        self.get_snakes(self.snake_heads, self.snake_lengths, )
        # self.set_snakes()

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
        self.board[pos[0], pos[1]] = val

    def get_snake(self, head_position, length, snake_num):

        self.snakes[snake_num][0] = head_position
        self.set_board_val(head_position, self.head_val)      
        next_eligible_root = head_position.copy()

        for i in range(length-1):            
            next_eligible_cells = self.get_all_eligible_neighbors(next_eligible_root)
            # print(f'recieved {next_eligible_cells}')
            next_eligible_root = next_eligible_cells[np.random.choice(np.arange(len(next_eligible_cells)))].squeeze()
            # print(f'selectred {next_eligible_root}')
            self.set_board_val(next_eligible_root, self.body_val)
            self.snakes[snake_num].append(next_eligible_root)
        
        # print(self.board)

    def get_all_eligible_neighbors(self, pos):
        # print(f'getting all eligible neighbors for {pos}')
        
        neighbors_h = np.array([-1, 0, 1]) # TODO: filter for border conditions
        neighbors_v = np.array([-1, 0, 1]) # TODO: filter for border conditions

        neighbors_h = neighbors_h + pos[0]
        neighbors_v = neighbors_v + pos[1]

        neighbors_h = self.position_mask(neighbors_h).reshape([-1,  1])
        neighbors_v = self.position_mask(neighbors_v).reshape([ 1, -1])
        
        positions_h, positions_v = np.broadcast_arrays(neighbors_h, neighbors_v)        
        positions = np.stack((positions_h, positions_v), axis=-1).reshape(-1, 2)[1::2]
        # print(positions)

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
        print(self.snakes)
        for row in range(self.size):
            row_str = ''
            for col in range(self.size):
                if row in [z[0] for z in self.food_positions] and col in [z[1] for z in self.food_positions]:
                    row_str += '|@ '
                elif row in [z[0] for z in self.snake_heads] and col in [z[1] for z in self.snake_heads]:
                    row_str += '|* '
                elif row in [z[0] for z in self.snakes[0]] and col in [z[1] for z in self.snakes[0]]:
                    row_str += '|++'
                else:
                    row_str += '|__'
            row_str += '|\n'
            string += row_str
        print(string)


    def move(self, dir=None):
        
        pass


board_size, snake_count, food_count = 10, 1, 1
game = Game(board_size, snake_count, food_count)