from game import *
from player import *
import numpy as np
import pygame

if __name__=="__main__":
    game = Game()

    player = Player(game=game)
    player.init_player()
    player.init_game.player.append(player)
    player.move('up')
    # player.init_game.display()
    # print(123)