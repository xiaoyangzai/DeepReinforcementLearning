#!/usr/bin/python

from human_player import human_player
from mcts_pure_player import MCTS_player 
from board import game

def main():
    g = game(width = 10,height = 10, n_in_row = 4)
    h_player = human_player(index = 1)
    mc_player = MCTS_player(index = 2,n_playout=1000)
    g.start_play(h_player,mc_player,is_shown = True)
    print("Game Over!!")
    return

if __name__ == "__main__":
    main()

