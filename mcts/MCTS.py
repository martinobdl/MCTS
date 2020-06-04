import numpy as np
import copy
from tqdm import tqdm_notebook as tqdm
import cloudpickle
import gym
from mcts.Nodes import DecisionNode, RandomNode


class MCTS:
    """
    Base class for MCTS based on Monte Carlo Tree Search for Continuous and Stochastic Sequential
    Decision Making Problems, Courtoux

    :param initial_obs: (int or tuple) initial state of the tree. Returned by env.reset().
    :param env: (gym env) game environment
    :param K: (float) exporation parameter of UCB
    """

    def __init__(self, initial_obs, env, K):
        self.env = env
        self.K = K
        self.root = DecisionNode(state=initial_obs, is_root=True)
        self._initialize_hash()

    def _initialize_hash(self):
        """
        Set the hash preprocessors of the state and the action, in order to make them hashable.
        """

        # action
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self._hash_action = lambda x: x
        elif isinstance(self.env.action_space, gym.spaces.Box):
            if self.__class__.__name__ == "MCTS":
                raise Exception("Cannot run vanilla MCTS on continuous actions")
            else:
                self._hash_action = lambda x: tuple(x)
        else:
            mex = "Action space has to be Discrete or Box, instead is {}".format(type(self.env.action_space))
            raise TypeError(mex)

        # observation
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            self._hash_space = lambda x: x
        elif isinstance(self.env.observation_space, gym.spaces.Box):
            self._hash_space = lambda x: tuple(x)
        else:
            mex = "Action space has to be Discrete or Box, instead is {}".format(type(self.env.observation_space))
            raise TypeError(mex)

    def update_decision_node(self, decision_node, random_node, hash_preprocess):
        """
        Return the decision node of drawn by the select outcome function.
        If it's a new node, it gets appended to the random node.
        Else returns the decsion node already stored in the random node.

        :param decision_node (DecisionNode): the decision node to update
        :param random_node (RandomNode): the random node of which the decison node is a child
        :param hash_preprocess (fun: gym.state -> hashable) function that sepcifies the preprocessing in order to make
        the state hashable.
        """

        if hash_preprocess(decision_node.state) not in random_node.children.keys():
            decision_node.father = random_node
            random_node.add_children(decision_node, hash_preprocess)
        else:
            decision_node = random_node.children[hash_preprocess(decision_node.state)]

        return decision_node

    def grow_tree(self):
        """
        Explores the current tree with the UCB principle until we reach an unvisited node
        where the reward is obtained with random rollouts.
        """

        decision_node = self.root
        internal_env = copy.copy(self.env)

        while (not decision_node.is_final) and decision_node.visits > 1:

            a = self.select(decision_node)

            new_random_node = decision_node.next_random_node(a, self._hash_action)

            (new_decision_node, r) = self.select_outcome(internal_env, new_random_node)

            new_decision_node = self.update_decision_node(new_decision_node, new_random_node, self._hash_space)

            new_decision_node.reward = r
            new_random_node.reward = r

            decision_node = new_decision_node

        decision_node.visits += 1
        cumulative_reward = self.evaluate(internal_env)

        while not decision_node.is_root:
            random_node = decision_node.father
            cumulative_reward += random_node.reward
            random_node.cumulative_reward += cumulative_reward
            random_node.visits += 1
            decision_node = random_node.father
            decision_node.visits += 1

    def evaluate(self, env):
        """
        Evaluates a DecionNode playing until an terminal node using the rollotPolicy

        :param env: (gym.env) gym environemt that describes the state at the node to evaulate.
        :return: (float) the cumulative reward observed during the tree traversing.
        """
        max_iter = 100
        R = 0
        done = False
        iter = 0
        while ((not done) and (iter < max_iter)):
            iter += 1
            a = env.action_space.sample()
            s, r, done, _ = env.step(a)
            R += r

        return R

    def select_outcome(self, env, random_node):
        """
        Given a RandomNode returns a DecisionNode

        :param: env: (gym env) the env that describes the state in which to select the outcome
        :param: random_node: (RandomNode) the random node from which selects the next state.
        :return: (DecisionNode) the selected Decision Node
        """
        new_state_index, r, done, _ = env.step(random_node.action)
        return DecisionNode(state=new_state_index, father=random_node, is_final=done), r

    def select(self, x):
        """
        Selects the action to play from the current decision node

        :param x: (DecisionNode) current decision node
        :return: (float) action to play
        """
        if x.visits <= 2:
            x.children = {a: RandomNode(a, father=x) for a in range(self.env.action_space.n)}

        def scoring(k):
            if x.children[k].visits > 0:
                return x.children[k].cumulative_reward/x.children[k].visits + \
                    self.K*np.sqrt(np.log(x.visits)/x.children[k].visits)
            else:
                return np.inf

        a = max(x.children, key=scoring)

        return a

    def best_action(self):
        """
        At the end of the simulations returns the most visited action

        :return: (float) the best action according to the number of visits
        """

        number_of_visits_children = [node.visits for node in self.root.children.values()]
        index_best_action = np.argmax(number_of_visits_children)

        a = list(self.root.children.values())[index_best_action].action
        return a

    def learn(self, Nsim, progress_bar=False):
        """
        Expand the tree and return the bet action

        :param: Nsim: (int) number of tree traversals to do
        :param: progress_bar: (bool) wether to show a progress bar (tqdm)
        """

        if progress_bar:
            iterations = tqdm(range(Nsim))
        else:
            iterations = range(Nsim)
        for _ in iterations:
            self.grow_tree()

    def forward(self, action, new_state):
        """
        If the env is determonostic we can salvage most of the tree structure.
        Advances the tree in the action taken if found in the tree nodes.

        :param action: (tuple)
        :param new_state: (tuple)
        """
        if self._hash_action(action) in self.root.children.keys():
            rnd_node = self.root.children[self._hash_action(action)]
            if len(rnd_node.children) > 1:
                self.root = DecisionNode(state=new_state, is_root=True)
            else:
                next_decision_node = np.random.choice(list(rnd_node.children.values()))
                if np.linalg.norm(next_decision_node.state-new_state) > 1e-3:
                    raise RuntimeWarning("The env is probably stochastic")
                else:
                    next_decision_node.father = None
                    self.root.children.pop(self._hash_action(action))
                    self.root = next_decision_node
                    self.root.is_root = True
        else:
            raise RuntimeWarning("Action taken: {} is not in the children of the root node.".format(action))

    def _collect_data(self):
        """
        Collects the data and parameters to save.
        """
        data = {
            "K": self.K,
            "root": self.root
        }
        return data

    def save(self, path=None):
        """
        Saves the tree structure as a pkl.

        :param path: (str) path in which to save the tree
        """
        data = self._collect_data()

        name = np.random.choice(['a', 'b', 'c', 'd', 'e', 'f']+list(map(str, range(0, 10))), size=8)
        if path is None:
            path = './logs/'+"".join(name)+'_'
        with open(path, "wb") as f:
            cloudpickle.dump(data, f)
        print("Saved at {}".format(path))

    def act(self):
        """
        Return the best action accoring to the maximum visits principle.
        """
        action = self.best_action()
        return action


if __name__ == "__main__":
    pass
