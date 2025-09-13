import json
import math
import os
import matplotlib.pyplot as plt
import networkx as nx
import logging

# TODO: check if all these max() things are actually needed.

# NOTE: Only instructions relevant for the calculations are explained. Logging instructions are excluded.
class Prophecy_Calculations:
    def __init__(self, data_file="training_data.json"):

        # # load training data (mind the correct format)
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"{data_file} not found.")
        with open(data_file, "r") as f:
            self.data = json.load(f)


        logging.basicConfig(
            filename="calculation_results.log",
            filemode="w",
            level=logging.INFO,
            format="%(message)s"    # %(asctime)s - %(levelname)s - 
        )

        self.log = logging.getLogger(__name__)

        self.is_root_call = 1
        self.random_temp = None
        self.valid_roots = []
        self.current_branch = None
        self.current_root = None

    def entropy(self, dataset, attr="FULL", subset_name="FULL DATASET"):

        total = len(dataset)
        if total == 0:
            return 0
        
        # read and calculate basic information for entropy calculations
        yes = sum(1 for d in dataset if d["decision"] == "yes")
        no = total - yes
        p_yes = yes / total if yes > 0 else 0
        p_no = no / total if no > 0 else 0

        # NOTE: formula from "util/HoT_Build-a-decision-tree.pdf | page 3 | section 1.1.3.b"
        entropy = (-(p_yes * math.log2(p_yes) if p_yes > 0 else 0)
               - (p_no * math.log2(p_no) if p_no > 0 else 0))
        
        if self.is_root_call == 1:
            if subset_name == "FULL DATASET":
                self.log.info("========== Section.1: Initial Data Calculations ==========\n\n")
                self.log.info("INITIAL IMPURITY for <%s> : %f", subset_name, entropy)
                self.log.info("")
            else:
                self.log.info("IMPURITY for group <%s> : %f", subset_name, entropy)
        self.random_temp = attr

        return entropy
    
    def get_root(self, dataset, used_attrs=None):

        # initialize "used_attrs" to be able to use it later
        if used_attrs is None:
            used_attrs = set()

        # note that the entropy always refers to the current dataset, so watch the call stack to understand the different values
        base_entropy = self.entropy(dataset=dataset)

        if self.is_root_call != 1:
            space = (self.is_root_call -1 ) * "\t" if self.is_root_call > 1 else ""
            self.log.info("%sEntropy for remaining dataset fulfilling <%s : %s>: %f", space, self.current_root, self.current_branch, base_entropy)

        best_gain = -1
        best_attr = None

        # retrieve all categories from the data (decision criteria)
        categories = [k for k in dataset[0].keys() if k != "decision" and k not in used_attrs]
        for attr in categories:
            if self.is_root_call == 1:
                self.log.info("\n########## Data for category <%s> ##########", attr.upper())

            # retrieve all parameters from the decision criterias
            parameters = set(d[attr] for d in dataset)
            if len(parameters) == 1:
                # if no split possible: no information can be gathered
                continue

            # calculate the weighted entropy for all categories
            weighted_entropy = 0
            for para in parameters:
                # sort by current "category: value" pairs
                subset = [d for d in dataset if d[attr] == para]
                # NOTE: formula from "util/HoT_Build-a-decision-tree.pdf | page 3 | section 1.1.3.c"
                weighted_entropy += (len(subset) / len(dataset) * self.entropy(dataset=subset, attr=attr, subset_name=f"{attr}={para}"))

            if self.is_root_call == 1:
                self.log.info("WEIGHTED AVERAGE IMPURITY for group <%s> : %f", attr, weighted_entropy)
            self.random_temp = attr

            # NOTE: formula from "util/HoT_Build-a-decision-tree.pdf | page 3 | section 1.1.3.d"
            gain = base_entropy - weighted_entropy

            if self.is_root_call == 1:
                self.log.info("GAIN for group <%s> : %f", attr, gain)
            self.random_temp = attr
            
            # set new "root" when better gain was found
            if gain > best_gain:        
                best_gain = gain
                best_attr = attr
                self.valid_roots = [(attr, gain)]

            elif gain == best_gain and best_attr is not None:
                self.valid_roots.append((attr, gain))
                # alphabetical tie break for categories with the same gain
                if attr > best_attr:
                    best_attr = attr

        self.valid_roots.sort(key=lambda x: x[1], reverse=True)
        if self.is_root_call == 1:
            self.log.info("\n\n========== Section.2: Decision Tree Build Process ==========")
            self.log.info("\n\nValid roots: %s\n\n", self.valid_roots)
            self.log.info("Chosen ROOT: <%s>", best_attr.upper())
        else: 
            space = (self.is_root_call -1 ) * "\t" if self.is_root_call > 1 else ""
            self.log.info("%sResult: <%s> -- Gain on remaining dataset: %f.", space, best_attr, best_gain)
        return best_attr

    def filter_data(self, dataset, root, subcategory):
        # returns all entries that fulfill "root" == "subcategory"
        return [category for category in dataset if category[root] == subcategory]
    
    def build_tree(self, dataset=None, used_attrs=None):
        
        # for the first call of this recursive function use the read data file
        if dataset is None:
            dataset = self.data
        
        # initialize "used_attrs" to be able to use it later
        if used_attrs is None:
            used_attrs = set()
        
        # labels are normally "yes" or "no", this line allows more labels than that.
        labels = set(d["decision"] for d in dataset)
        
        # if only one label remains: the current attribute resembles a leaf
        if len(labels) == 1:
            space = (self.is_root_call -1 ) * "\t" if self.is_root_call > 1 else ""
            self.log.info("%sResult: (%s)", space, next(iter(labels)).upper())
            return labels.pop()

        # get all categories that are not used yet
        categories = [k for k in dataset[0].keys() if k != "decision" and k not in used_attrs]
        
        if not categories:
            # choose most frequent decision if no categories are left
            return max(labels, key=lambda x: sum(1 for d in dataset if d["decision"] == x))
        
        # define the root for the current "sub-tree"
        root = self.get_root(dataset=dataset, used_attrs=used_attrs)
        
        if root is None:
            # choose most frequent decsion if no root category can be found
            return max(labels, key=lambda x: sum(1 for d in dataset if d["decision"] == x))

        # create the base of the sub-tree in json format
        tree = {root: {}}

        # parameters of the current category
        parameters = set(d[root] for d in dataset)
        
        for para in parameters:
            # get subset sorted by "category: parameter"
            subset = self.filter_data(dataset=dataset, root=root, subcategory=para)
        
            if not subset:
                # choose the most frequent decision if subset is empty
                tree[root][para] = max(labels, key=lambda x: sum(1 for d in dataset if d["decision"] == x))
            else:
                # add the category to the already used categories
                new_used = used_attrs | {root}

                

                self.is_root_call = self.is_root_call + 1
                space = (self.is_root_call -1 ) * "\t" if self.is_root_call > 1 else ""
                self.log.info("\n%sBranch: <%s : %s>", space, root, para)
                self.current_branch = para
                self.current_root = root
                
                # add another subtree to the root by calling the "build_tree" function with a reduced data_set
                tree[root][para] = self.build_tree(dataset=subset, used_attrs=new_used)

                self.is_root_call = self.is_root_call - 1   

        if self.is_root_call == 1:
            self.log.info("\nAll branches ended in a leaf. No more possibilities and tree is finished!")   

        return tree
    
    def hierarchy_pos(self, G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        
        if not nx.is_tree(G):
            self.log.error("Given graph is not a tree!")
            raise TypeError("Graph is not a tree")
        
        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))
            else:
                root = list(G.nodes)[0]


        def _hierarchy_pos(G, root, left, right, vert_loc, pos=None, parent=None):
            
            if pos is None:
                pos = {}
            # set root position in the plot (mid screen)
            pos[root] = ((left + right) / 2, vert_loc)
            # all branches from root
            children = list(G.successors(root))
            # if rooot is not a leaf
            if len(children) != 0:
                # evenly distribute nodes across the screen
                dx = (right - left) / len(children)
                nextx = left
                # recursively repeat for every subtree
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G, child, nextx - dx, nextx, vert_loc - vert_gap, pos, root)
            return pos
        return _hierarchy_pos(G=G, root=root, left=0, right=width, vert_loc=vert_loc, pos=None, parent=None)
    
    def plot_tree(self, tree):

        G = nx.DiGraph()

        def add_nodes_edges(subtree, parent=None, edge_label=""):
            # if subtree is a real subtree
            if isinstance(subtree, dict):
                # go through all nodes and their branches
                for node, branches in subtree.items():
                    # check if node is the first node
                    unique_node = node if parent else node
                    if parent is not None:
                        G.add_edge(parent, unique_node, label=edge_label)
                    # go recursively through the "subtrees"
                    for subcat, child in branches.items():
                        if isinstance(child, dict):
                            add_nodes_edges(subtree=child, parent=unique_node, edge_label=str(subcat))
                        else:
                            # you need to surpass the leaf-label-process due to NetworkX.DiGraph constraints for unique leafs
                            leaf_name = f"{unique_node}_{subcat}_{child}"
                            G.add_edge(unique_node, leaf_name, label=str(subcat))
                            G.nodes[leaf_name]["is_leaf"] = True
                            G.nodes[leaf_name]["label"] = child.upper()
            # if we reached the end of the tree an the subtree is actually a leaf
            else:
                G.add_node(subtree, is_leaf=True)
                if parent:
                    G.add_edge(parent, subtree, label=edge_label)


        add_nodes_edges(subtree=tree)

        # retrieve root to build tree structure from here
        root = list(tree.keys())[0] if isinstance(tree, dict) else None
        # get the positions for the leafs
        pos = self.hierarchy_pos(G=G, root=root)
        
        # plot the graph
        plt.figure(figsize=(12, 8))

        # get labels for all nodes and convert to uppercase
        labels = {n: G.nodes[n].get("label", n).upper() for n in G.nodes()}
        
        # finally draw the raw graph
        nx.draw(
            G, pos, with_labels=True, labels=labels, node_size=2000,
            # choose color dependend on node being a normal node or a leaf
            node_color=["lightgreen" if G.nodes[n].get("is_leaf") else "lightblue" for n in G.nodes()],
            font_size=9
        )

        # draw the edge labels connecting the nodes
        edge_labels = nx.get_edge_attributes(G, "label")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        

        plt.title("Final Decision Tree")
        # do not need axises
        plt.axis("off")
        plt.show()


def main():

    # create an instance and pass the training data (default is "training_data.json")
    prophecy = Prophecy_Calculations(data_file="training_data.json")
    
    # calculate and build the tree
    tree = prophecy.build_tree()
    json.dumps(obj=tree, indent=2)
    prophecy.plot_tree(tree=tree)


if __name__ == "__main__":
    main()