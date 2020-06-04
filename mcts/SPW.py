import numpy as np
from mcts.MCTS import MCTS


class SPW(MCTS):
    """
    Simple Progressive Widening trees based on Monte Carlo Tree Search for Continuous and
    Stochastic Sequential Decision Making Problems, Courtoux

    :param alpha: (float) the number of children of a decision node are always greater that v**alpha,
        where v is the number of visits to the current decision node
    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: exploration parameter of UCB
    """

    def __init__(self, alpha, initial_obs, env, K):
        super(SPW, self).__init__(initial_obs, env, K)

        self.alpha = alpha

    def select(self, x):
        """
        Selects the action to play from the current decision node. The number of children of a DecisionNode is
        kept finite at all times and monotonic to the number of visits of the DecisionNode.

        :param x: (DecisionNode) current decision node
        :return: (float) action to play
        """
        if x.visits**self.alpha >= len(x.children):
            a = self.env.action_space.sample()

        else:

            def scoring(k):
                if x.children[k].visits > 0:
                    return x.children[k].cumulative_reward/x.children[k].visits + \
                        self.K*np.sqrt(np.log(x.visits)/x.children[k].visits)
                else:
                    return np.inf

            a = max(x.children, key=scoring)

        return a

    def _collect_data(self):
        """
        Collects the data and parameters to save.
        """
        data = {
            "K": self.K,
            "root": self.root,
            "alpha": self.alpha
        }
        return data
