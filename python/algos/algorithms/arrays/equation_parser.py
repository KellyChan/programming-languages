"""
Parsing an equation in a string/array.
"""

import compiler

def parser_str_compiler(equation_str):
    return compiler.parse(equation_str)

#--------------------------------------------------#

import parser


def parser_str_expr(equation_str):
    return parser.expr(equation_str).compile()
    
#-------------------------------------------------#

class Tree(object):

    def __init__(self):
        self.left = None
        self.right = None
        self.data = None

if __name__ == '__main__':
   
    equation_str = "sin(x)*x**2"

    print "parser_str_compiler: %s" % parser_str_compiler(equation_str)
    print "parser_str_expr: %s" % parser_str_expr(equation_str)

    tree = Tree()
    tree.data = "root"
    tree.left = Tree()
    tree.left.data = "left"
    tree.right = Tree()
    tree.right.data = "right"
    print "tree: %s" % tree
    print "tree.data: %s" % tree.data
    print "tree.left: %s" % tree.left
