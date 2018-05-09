#!/usr/bin/python

from human_player import human_player
from mcts_pure_player import MCTS_player 
from board import game

def main():
    g = game(width = 5,height = 5, n_in_row = 3)
    h_player = human_player(index = 1)
    #mc_player1 = MCTS_player(index = 1,n_playout=5000)
    mc_player2 = MCTS_player(index = 2,n_playout=10000)
    g.start_play(h_player,mc_player2,is_shown = True)
    print("Game Over!!")
    return

if __name__ == "__main__":
    main()

