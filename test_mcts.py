import unittest
from mcts import *
import numpy 

class TestMCTS(unittest.TestCase):

    def setUp(self):
    
        l = 1
        w = 0.7
        a = 0.7
        h = 1
        n = 10

        def reward(x):
            """
            x: Decision_Node obj
            return: float reward
            """
            if x.state[0]<l:
                return a
            if x.state[0]>l+w:
                return h
            return 0

        def transition(x, a):
            """
            takes a decision node and an action and outputs another
            decision node based on the transition dynamics

            x: state_object
            a: numeric
            """
            x_new = Decision_Node(
                state = (x.state[0]+a, x.state[1]+1),
                father = None,
                is_root = False,
                is_final = (x.state[1]+1 == 2))

            return x_new

        def generative_model(random_node):
            """
            state: state_variable
            a: numeric
            """
            state_new = transition(random_node.father, random_node.action)
            r = reward(state_new)
            return (state_new, r)

        def rollout_policy(x):
            """
            x: numeric
            return a random legal action
            """
            if DISCRETE:
                action = np.random.choice(n+1)/n
            else:
                action = np.random.uniform()
            return action

        def action_sampler(state):
            if DISCRETE:
                action = np.random.choice(n+1)/n
            else:
                action = np.random.uniform()
            return action

        self.x = Decision_Node((0,0), father=None, is_root = True)

        self.tree_mcts = MCTS(initial_state = self.x,
                        K = 3,
                        generative_model = generative_model,
                        rollout_policy = rollout_policy,
                        action_sampler = action_sampler)

        self.tree_SPW = SPW(alpha = 0.7,
                        initial_state = self.x,
                        K = 3,
                        generative_model = generative_model,
                        rollout_policy = rollout_policy,
                        action_sampler = action_sampler)

        self.tree_DPW = DPW(alpha = 0.7,
                        beta = 0.7,
                        initial_state = self.x,
                        K = 3,
                        generative_model = generative_model,
                        rollout_policy = rollout_policy,
                        action_sampler = action_sampler)
        
    def tearDown(self):
        pass

    def test_MCTS_select_outcome(self):
        x =  Random_Node(state = (0,0),
                        action = 0.5,
                        father = self.x)
        _,r = self.tree_mcts.select_outcome(x)
        self.assertEqual(r,0.7)

        x =  Random_Node(state = (0,0),
                        action = 1.8,
                        father = self.x)
        _,r = self.tree_mcts.select_outcome(x)
        self.assertEqual(r,1)

if __name__ == "__main__":
    unittest.main()
