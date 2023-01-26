from game import *
from player import *
from ga_agent_v2_3 import *
import numpy as np
import pygame

move_dict = {'w': 'up', 's': 'down', 'a': 'left', 'd': 'right'}

if __name__=="__main__":
    base_path = "weights/genetic_algorithm/"
    population_name = 'test_2_3'


    game, player = initialize_game_with_player(set_seed=False)
    generation = 99
    all_brains = load_weights(population_name, base_path, generation)
    brain = generate_brain(all_brains[0], layers)