import numpy as np
from mcts.SPW import SPW
from mcts.Nodes import DecisionNode


class DPW(SPW):
    """
    Double Progressive Widening trees based on MCTS for Continuous and
        Stochastic Sequential Decision Making Problems, Courtoux.

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha,
        where v is the number of visits to the current decision node
    :param beta: (float) the number of outcomes of a random node is grater that v**beta,
        where v is the number of visits of the random node
    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: exploration parameter of UCB
    """
    def __init__(self, alpha, beta, initial_obs, env, K):

        super(DPW, self).__init__(alpha, initial_obs, env, K)

        self.beta = beta

    def select_outcome(self, env, random_node):
        """
        The number of outcomes of a RandomNode is kept fixed at all times and increasing
        in the number of visits of the random_node

        :param: random_node: (RandomNode) random node from which to select the next state
        :return: (DecisionNode, float) return the next decision node and reward
        """

        if random_node.visits**self.beta >= len(random_node.children):
            new_state_index, r, done, _ = env.step(random_node.action)
            return DecisionNode(state=new_state_index, father=random_node, is_final=done), r

        else:
            unnorm_probs = [child.visits for child in random_node.children.values()]
            probs = np.array(unnorm_probs)/np.sum(unnorm_probs)

            chosen_state = np.random.choice(list(random_node.children.values()), p=probs)
            return (chosen_state, chosen_state.reward)
