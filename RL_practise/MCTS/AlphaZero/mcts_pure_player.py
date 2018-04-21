#!/usr/bin/python

import numpy as np
import copy
import time
from operator import itemgetter

def rollout_policy_fn(board):
    """
        A coarse, fast version of policy_fn used in the rollout phase
    """
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables,action_probs)

def policy_value_function(board):
    """
        A function that takes in a state and outputs a list of (action, probability)
        tuples and a score for the state
    """
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables,action_probs)

class TreeNode(object):
    """
        A node in the MCTS tree. Each node keeps track of its own value Q,
        prior probabiliyu P, and upper confident bound U.

    """
    def __init__(self,parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p 

    def expand(self,action_priors):
        """
            Expand tree by creating new children.
            action_priors: a list of tuples of actions and their prior probability according to the policy function.

        """

        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self,prob)


    def select(self, c_puct):
        """
            Select action among children that gives maximum action value Q plus bonus u(p)
            Return: A tuple of (action, next_node)
        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self,c_puct):
        """
            Calculate the UCB. Return the Q + U
        """
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def update(self,leaf_value):
        """
            Update node values from leaf evaluation.
            leaf_value: the value of subtree evaluation from the current player's perspective
        """
        self._n_visits += 1

        #update Q
        self._Q += 1.0 * (leaf_value - self._Q)/self._n_visits
        
    def update_recursive(self,leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """
            Check if leaf node
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS_pure(object):
    """
        A simple implementation of Monte Carlo Tree Seatch
    """
    def __init__(self,policy_value_fu,c_puct = 5,n_playout = 10000):
        """
            policy_value_fn: a function that takes a board state as input and outputs a list of (action, probability) tuples and a score in [-1,1].
        """
        self._root = TreeNode(None,1.0)
        self._policy = policy_value_fu
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self,state,is_shown = False):
        """
            Run a single playout from the root to the leaf, getting a value at the leaf and back up through their parents. 
            State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if is_shown:
                state.show_board()
            if node.is_leaf():
                break
            #select policy in tree is using the UCB
            action, node = node.select(self._c_puct)
            state.do_move(action)


        #Step 2: the select policy out of tree is using the random policy(policy_value_fu) untile the game is over
        action_probs = self._policy(state)
        win_flag,winner = state.is_game_over()
        if win_flag == 0:
            node.expand(action_probs)

        leaf_value = self._evaluate_rollout(state,is_shown=is_shown)
        #Step 3: back up 
        node.update_recursive(leaf_value)

        return True if leaf_value == 1 else False 

    def _evaluate_rollout(self, state, max_steps = 1000,is_shown=False):
        """
            Use the rollout policy to play untile the end of the game, returning +1 if the current_player wins, -1 if the opponent wins and 0 if it is a tie.
        """
        player = state.get_current_player()
        for i in range(max_steps):
            if is_shown:
                state.show_board()
                time.sleep(1)
            win_flag,winner = state.is_game_over()
            if win_flag != 0:
                break
            #policy out of search tree: random
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs,key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            print("Warning: rollout reached moce limit!")

        if state.win_flag == 0 or state.win_flag == 2:
            #tie
            return 0
        else:
            reward = 1 if state.winner == player else -1 
            return reward

    def get_move(self,state):
        """
            Runs all playouts  and returns the most visited action.
        """
        is_shown = False
        win_count = 0
        for n in range(self._n_playout):
            #playing games repeatly and selecting the most visited action.
            #keep the current board state from modifing in each simulation of the games
            state_copy = copy.deepcopy(state)
            #playing a game under the simulation
            leaf_value = self._playout(state_copy,is_shown)
            if leaf_value == 1:
                win_count += 1
            if n == self._n_playout - 1:
                is_shown = True
            else:
                is_shown = False 

        temp = [[item[0],item[1]._n_visits] for item in self._root._children.items()]
        temp = sorted(temp,key=lambda item: item[1],reverse=True)[:5]
        print("[%d]Thinking policy: "%n,temp)
        print("Win probability: %d-%d"%(win_count,self._n_playout))
        h,w = state.move_to_location(temp[0][0])
        print("MCTS player will move to location[%d,%d]"%(h,w))
        time.sleep(10)
        #After all simulation games, return the most visited action from the current state
        return max(self._root._children.items(),key=lambda act_node:act_node[1]._n_visits)[0]
   
    def update_search_treee_after_move(self, last_move):
        """
            update the search tree, when we perform the move on the current state.
            After performing move under the state, the next state becomes the new root node;
            and the subtree below this next state is retained along with all statistics, 
            while the remainder of the tree is dicarded.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None,1.0)

    def __str__(self):
        return "MCTS"

class MCTS_player(object):
    """
        AI player based on MCTS
    """
    def __init__(self,index,c_puct=5, n_playout=2000):
        self.mcts = MCTS_pure(policy_value_function,c_puct,n_playout=n_playout)
        self.index = index


    def set_player_index(self,index):
        self.player = index

    def reset_player(self):
        self.mcts.update_search_treee_after_move(-1)

    def get_action(self, board):
        availables_move = board.availables
        if len(availables_move) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_search_treee_after_move(-1)
            return move
        else:
            print("Warning: the board is full")

    def __str__(self):
        return "MCTS player index[%d]"%self.player
