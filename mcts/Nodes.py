

class DecisionNode:
    """
    The Decision Node class

    :param state: (tuple) defining the state
    :param father: (RandomNode) The father of the Decision Node, None if root node
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state=None, father=None, is_root=False, is_final=False):
        self.state = state
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.reward = 0
        self.father = father
        self.is_root = is_root

    def add_children(self, random_node, hash_preprocess=None):
        """
        Adds a RandomNode object to the dictionary of children (key is the action)

        :param random_node: (RandomNode) add a random node to the set of children visited
        """
        if hash_preprocess is None:
            def hash_preprocess(x):
                return x

        self.children[hash_preprocess(random_node.action)] = random_node

    def next_random_node(self, action, hash_preprocess=None):
        """
        Add the random node to the children of the decision node if note present. Otherwise it resturns the existing one

        :param action: (float) the actiuon taken at the current node
        :return: (RandomNode) the resutling random node
        """

        if hash_preprocess is None:
            def hash_preprocess(x):
                return x

        if hash_preprocess(action) not in self.children.keys():
            new_random_node = RandomNode(action, father=self)
            self.add_children(new_random_node, hash_preprocess)
        else:
            new_random_node = self.children[hash_preprocess(action)]

        return new_random_node

    def __repr__(self):
        s = ""
        for k, v in self.__dict__.items():
            if k == "children":
                pass
            elif k == "father":
                pass
            else:
                s += str(k)+": "+str(v)+"\n"
        return s


class RandomNode:
    """
    The RandomNode class defined by the state and the action, it's a random node since the next state is not yet defined

    :param action: (action) taken in the decision node
    :param father: (DecisionNode)
    """

    def __init__(self, action, father=None):
        self.action = action
        self.children = {}
        self.cumulative_reward = 0
        self.visits = 0
        self.father = father

    def add_children(self, x, hash_preprocess):
        """
        adds a DecisionNode object to the dictionary of children (key is the state)

        :param x: (DecisinNode) the decision node to add to the children dict
        """
        self.children[hash_preprocess(x.state)] = x

    def __repr__(self):
        mean_rew = round(self.cumulative_reward/(self.visits+1), 2)
        s = "action: {}\nmean_reward: {}\nvisits: {}".format(self.action, mean_rew, self.visits)
        return s
