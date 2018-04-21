#!/usr/bin/python

from __future__ import print_function
import numpy as np
import os
from human_player import human_player
import time

class Board(object):
    """board for game"""

    def __init__(self,**kwargs):
        self.width = int(kwargs.get('width',8))
        self.height = int(kwargs.get('height',8))
        #location:player
        self.states = {}
        self.win_flag = 0
        self.winner = 0
        self.n_in_row = int(kwargs.get('n_in_row',5))
        #player1 and player2
        self.players = [1,2]
        self.start_player = int(kwargs.get('start_player',0))
        return

    def init_board(self,start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width or height can not be less than {}',format(self.n_in_row))
        self.current_player = self.players[start_player]
        self.availables = list(range(self.width*self.height))
        self.states = {}
        self.last_move = -1
        return

    def move_to_location(self,move):
        '''
            3*3 board's moves like:
            6 7 8
            3 4 5
            0 1 2
            eg: move 5's location is (1,2)
        '''
        h = move // self.height
        w = move % self.width
        return [h,w]
    def location_to_move(self,h,w):
        move = h * self.width  + w 
        return move
    
    def current_state(self):
        '''
            return the board state
            state shape: 4*width*height
            state: the player state, the next player state, the next player state of last location; is first player.
        '''
        square_state = np.zeros((4,self.width,self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]

            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_curr % self.height] = 1.0

            #last move location
            square_state[2][self.last_move // self.width,self.last_move % self.height] = 1.0

        if len(self.states) % 2 == 0:
            square_state[3][:,:] = 1.0
        return square_state

    def do_move(self,move):
        if move not in self.availables:
            self.win_flag = -1
            return self.win_flag,self.winner
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.last_move = move
        self.win_flag = self.is_winner()
        if self.win_flag == 1:
            self.winner = self.current_player 
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1]
        return

    def is_game_over(self):
            return self.win_flag,self.winner 

    def up_down_count(self):
        '''
            3*3 board's moves like:
            6 7 8
            3 4 5
            0 1 2
            eg: move 5's location is (1,2)
        '''
        count = 1
        #count up_down direction
        h,w = self.move_to_location(self.last_move)
        while h < self.height:
            temp_up_move = self.location_to_move(h+1,w)
            temp_move = self.location_to_move(h,w)
            if temp_up_move in self.states and self.states[temp_up_move] == self.states[temp_move]:
                count += 1
                h += 1
            else:
                break
        h,w = self.move_to_location(self.last_move)
        while h > 0:
            temp_down_move = self.location_to_move(h-1,w)
            temp_move = self.location_to_move(h,w)
            if temp_down_move in self.states and self.states[temp_down_move] == self.states[temp_move]:
                count += 1
                h -= 1 
            else:
                break
        return count 
    def left_right_count(self):
        count = 1
        h,w = self.move_to_location(self.last_move)
        while w > 0:
            temp_left_move = self.location_to_move(h,w - 1) 
            temp_move = self.location_to_move(h,w) 
            if temp_left_move in self.states and self.states[temp_left_move] == self.states[temp_move]:
                count += 1
                w -= 1
            else:
                break
        h,w = self.move_to_location(self.last_move)
        while w < self.width:
            temp_right_move = self.location_to_move(h,w+1) 
            temp_move = self.location_to_move(h,w) 
            if temp_right_move in self.states and self.states[temp_right_move] == self.states[temp_move]:
                count += 1
                w += 1
            else:
                break
        return count
    
    def left_up_to_right_down(self):
        count = 1
        h,w = self.move_to_location(self.last_move)
        while h < self.height and w > 0:
            temp_left_up_move = self.location_to_move(h + 1, w - 1)
            temp_move = self.location_to_move(h,w)
            if temp_left_up_move in self.states and self.states[temp_left_up_move] == self.states[temp_move]:
                count += 1
                h += 1
                w -= 1
            else:
                break

        h,w = self.move_to_location(self.last_move)
        while h > 0 and w < self.width:
            temp_right_down_move = self.location_to_move(h - 1, w + 1)
            temp_move = self.location_to_move(h,w)
            if temp_right_down_move in self.states and self.states[temp_right_down_move] == self.states[temp_move]:
                count += 1
                h -= 1
                w += 1
            else:
                break
        return count
    
    def right_up_to_left_down(self):
        count = 1
        h,w = self.move_to_location(self.last_move)
        while h < self.height and w < self.width:
            temp_right_up_move = self.location_to_move(h + 1, w + 1)
            temp_move = self.location_to_move(h,w)
            if temp_right_up_move in self.states and self.states[temp_right_up_move] == self.states[temp_move]:
                count += 1
                h += 1
                w += 1
            else:
                break

        h,w = self.move_to_location(self.last_move)
        while h > 0 and w > 0:
            temp_left_down_move = self.location_to_move(h - 1, w - 1)
            temp_move = self.location_to_move(h,w)
            if temp_left_down_move in self.states and self.states[temp_left_down_move] == self.states[temp_move]:
                count += 1
                h -= 1
                w -= 1
            else:
                break

        return count

    def get_current_player(self):
        return self.current_player
    
    def show_board(self):
        os.system("clear")
        char_type = {1:"X",2:"O"}
        for _ in range(self.width):
            print("* ",end='')
        print("* *")
        for i in range(self.height):
            print("* ",end='')
            for j in range(self.width): 
                move = self.location_to_move(i,j)
                if move in self.states:
                    print(char_type[self.states[move]],"",end='')
                else:
                    print("- ",end='')
            print("*")
        for _ in range(self.width):
            print("* ",end='')
        print("* *")

    
    def is_winner(self):
        if len(self.states) == self.width * self.height:
            #pass
            print("No one has win!!")
            return 2
        #Determine whether to win based on the position of the last move 
        #if len(self.states) < 2*(self.n_in_row - 1) or self.last_move == -1:
        #    return 0 
        #count up_down direction
        count = self.up_down_count() 
        #print("up to down count : %d"%count)
        if count >= self.n_in_row:
            return 1 
        count = self.left_right_count() 
        #print("left to right count : %d"%count)
        if count >= self.n_in_row:
            return 1 
        count = self.left_up_to_right_down() 
        #print("left up to right down count : %d"%count)
        if count >= self.n_in_row:
            return 1 

        count = self.right_up_to_left_down()
        #print("right up to left down count : %d"%count)
        if count >= self.n_in_row:
            return 1 
        return 0 

class game():
    def __init__(self,**kwargs):
        self.width = int(kwargs.get('width',8))
        self.height = int(kwargs.get('height',8))
        #location:player
        self.states = {}
        self.n_in_row = int(kwargs.get('n_in_row',5))
        #player1 and player2
        self.start_player = int(kwargs.get('start_player',0))
        self.board = Board(width = self.width,height = self.height,n_in_row = self.n_in_row,start_player = self.start_player)
        return

    def start_play(self,player1,player2,is_shown = False):
        self.board.init_board()
        p1, p2 = self.board.players
        player1.set_player_index(p1)
        player2.set_player_index(p2)
        players = {p1: player1,p2: player2}
        print("Index of player1: %d\tIndex of player2: %d"%(p1,p2))
        win_flag = -1 
        winner = -1
        while True:
            if is_shown:
                self.board.show_board()
            current_player_index = self.board.current_player
            current_player = players[current_player_index] 
            print("Waiting for %s to move...."%current_player)
            #play output the move
            move = current_player.get_action(self.board)

            #exacute the move
            self.board.do_move(move)
            if is_shown:
                self.board.show_board()

            win_flag, winner = self.board.is_game_over()
            if win_flag < 0:
                h,w = self.board.move_to_location(move)
                print("location [%d,%d] is invaled! Try again!"%(h,w))
                time.sleep(2)
            if win_flag > 0:
                break

        if win_flag == 1:
            print("========* [%d] player wins the game! *========"%winner)
        else:
            print("No Winner!!")

    def start_self_play(self,player,is_shown,temp=1e-3):
        self.board.init_board()
        players = {p1: player1,p2: player2}
        win_flag = -1 
        winner = -1
        states, mcts_probs, players_order = [], [], []
        rewards_z = None
        while True:
            current_player_index = self.board.current_player
            current_player = players[current_player_index] 
            move, move_probs = player.get_action(self.board,temp=temp,return_prob = True) 

            #store the state
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            players_order.append(current_player_index)

            #perform a move
            self.board.do_move(move)
            if is_shown:
                self.board.show_board()
            if self.win_flag < 0:
                print("location [%d,%d] is invaled! Try again!"%(h,w))
            if win_flag > 0:
                break

        rewards_z = np.array(len(players_order))
        if self.win_flag == 1:
            print("========* [%d] player wins the game! *========"%winner)
            #reward of each move
            rewards_z[np.array(players_order) == self.winner] = 1.0
            rewards_z[np.array(players_order) != self.winner] = -1.0
            #reset MCTS root node
            player.reset_player()
        else:
            print("No Winner!!")
        return self.win_flag,self.winner,zip(states,mcts_probs,players_order)


def main():
    g = game(width = 10,height = 10, n_in_row = 3)
    player1 = human_player(1)
    player2 = human_player(2)
    g.start_play(player1,player2,is_shown = True)
    print("Game Over!!")
    return

if __name__ == "__main__":
    main()
