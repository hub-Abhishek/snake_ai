from game import *
from player import *
import numpy as np
import pygame

move_dict = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}

if __name__=="__main__":
    game = Game()

    player = Player(game=game)
    player.init_player()
    game.player.append(player)
    player.init_game.player.append(player)
    while player.aa_aa_aa_aa_stayin_alive:
        dir = input('dir:')
        dir = move_dict.get(dir, None)
        player.move(dir)
        print(player.init_game.snakes)
        print(player.init_game.food_positions)