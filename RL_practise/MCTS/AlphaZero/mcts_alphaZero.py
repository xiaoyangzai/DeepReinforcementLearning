#!/usr/bin/python

import numpy as np
import copy

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """
        A node in the MCTree.
        Each node keeps track of its own value Q, prior probablity P and its upper confident bound value U
    """

    def __init__(self,parent, prior_p):
        self.parent = parent

        #a map from action to TreeNode
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self,action_priors):
        """
            Expand tree by creating new children
            action_priors: alist of tuples of actions with their prior probality according to the policy function
        """
        for action, prob in action_priors:
            self._children[action] = TreeNode(self,prob)

    def select(self, c_puct):
        """
            Select action among children that gives maximum action value Q plus the bonus U
            a_t = argmax_a(Q(s_t,a) + U(s_t,a))
            Return: A tuple of (action, next_node)

        """
        return max(self._children.items(), key=lambda act_node: act_node[1].get_value(c_puct))

    def get_value(self, c_puct):
        """
            Calculate the UCB value of the node.
            Q_ucb = Q(s,a) + U(s,a)
            U(s,a) = c_puct * P(s,a) * sqrt(N(s,*)) / (1 + N(s,a))

        """
        self._u = c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits)
        return self._u + self._Q

    def update(self, leaf_value):
        """
            Update the node values from leaf evaluation.
            leaf_value: the value of subtree is evaluated by policy-value function with the current player's perspective. 
        """
        #update visists count
        self._n_visits += 1

        #update mean Q value 
        #u_k = (sum fron i = 1 to k x_i) over k 
        #    = u_{k-1} +  (x_{k-1} - u_{k-1})over k
        self._Q = self._Q + (leaf_value - self._Q) / self._n_visits

    def update_recursive(self,leaf_value):
        """
            Update all node where t <= L.
        """
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """
            Check if leaf node (i.e. no children)
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None
        

class MCTS(object):
    """
        An implementation of Monte Carlo Tree Search.
    """

    def __init__(self, policy_value_network, c_puct) = 5, n_playout = 10000):
        """
            policy_value_network: the network that takes the board state as input and  takes all actions' probilities(a list of tuples(action, Probility)) and the value of the state as the output. 
            c_puct : a number in (0,inf) that controls the level of the exploration. Higher value means more exploration 
        """
        self._root = TreeNode(None,1.0)
        self._policy = policy_value_network
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self,state):
        """
            Run a single playout from the root to the leaf, getting the value of the leaf node calculated by policy_value network and backup to all nodes from leaf node to root  
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            #in-tree of MC tree, using the puct to select next move
            action, node = node.select(self._c_puct)
            win_flag,winner = state.do_move(action)
            if win_flag > 0:
                break

        action_probs, leaf_values = self._policy(state)
        if win_flag == 0:
            node.expand(action_probs)
        else if win_flag == 1:
            leaf_value = 1.0 if winner == state.get_current_player() else -1.0
        else if win_flag == 2:
            leaf_value = 0.0
        node.update_recursive(leaf_value)

