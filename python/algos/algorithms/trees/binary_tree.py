"""
Binary Tree
"""

class Node:

    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

class BinaryTree:

    def __init__(self):
        self.root = None

    def get_root(self):
        return self.root

    def add_node(self, value):
        if (self.root == None):
            self.root = Node(value)
        else:
            self._add_child(value, self.root)

    def _add_child(self, value, node):
        if (value < node.value):
            if (node.left != None):
                self._add_child(value, node.left)
            else:
                node.left = Node(value)
        else:
            if (node.right != None):
                self._add_child(value, node.right)
            else:
                node.right = Node(value)

    def find(self, value):
        if (self.root != None):
            return self._find_child(value, self.root)
        else:
            return None

    def _find_child(self, value, node):
        if (value == node.value):
            return node
        elif (value < node.value and node.left != None):
            self._find_child(value, node.left)
        elif (value > node.value and node.right != None):
            self._find_child(value, node.right)

    def delete_tree(self):
        # garbage collector will do this for us.
        self.root = None

    def print_tree(self):
        if (self.root != None):
            self._print_tree(self.root)

    def _print_tree(self, node):
        if node:
            self._print_tree(node.left)
            print str(node.value) + ' '
            self._print_tree(node.right)


if __name__ == '__main__':

    binary_tree = BinaryTree()
    #print binary_tree.get_root()

    root = binary_tree.add_node(2)
    #print binary_tree.get_root().value

    binary_tree.add_node(3)
    #print binary_tree.print_tree()

    binary_tree.add_node(1)
    #print binary_tree.print_tree()

    binary_tree.add_node(10)
    print binary_tree.print_tree()
    print binary_tree.find(3)
