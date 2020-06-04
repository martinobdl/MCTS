from graphviz import Digraph
import pickle
import argparse
import os


def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise NotADirectoryError(string)


def name(node):
    return node.__repr__().replace(") (", ")\n(")


def recursive_graph(node):

    def inner_fnct(node):
        g.node(str(node.__hash__()), name(node))
        for k, v in node.children.items():
            g.node(str(v.__hash__()), name(v), shape='square')
            g.edge(str(node.__hash__()), str(v.__hash__()))
            for k2, v2 in v.children.items():
                g.node(str(v2.__hash__()), label=name(v2))
                g.edge(str(v.__hash__()), str(v2.__hash__()))
                inner_fnct(v2)

    inner_fnct(node)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=dir_path)

    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data = pickle.load(f)

    root = data["root"]

    g = Digraph()

    recursive_graph(root)
    file_name = os.path.basename(args.path)

    g.render('img/'+file_name+'.gv', view=True)
