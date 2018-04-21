#!/usr/bin/python
class human_player():
    def __init__(self,index):
        self.index = index 
        print("Human play[%d] has creator!!"%index)

    def get_action(self,board = None):
        if self.index < 0:
            print("The player havn't index, please set index for this player!")
            return -1
        h,w = map(int,raw_input("please [%d] player input loaction[x,x]:"%board.current_player).split(','))
        move = board.location_to_move(h,w)
        return move 

    def set_player_index(self,index):
        self.index = index
        return

    def __str__(self):
        return "Human player index[%d]"%self.index
