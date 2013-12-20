# Optimization Phase

def optimize(tree): # Expression trees only
    etype = tree[0]
    if etype == "binop":
        a = tree[1]
        op = tree[2]
        b = tree[3]
        if op == "*" and b == ("number","1"):
            return a
        elif op == "*" and b == ("number","0"):
            return b
        elif op == "+" and b == ("number","0"):
            return a
        # QUIZ: It only handles A * 1
        # Add in support for A * 0  and A + 0

        return tree
    

print optimize(("binop",("number","5"),"*",("number","1"))) == ("number","5")
print optimize(("binop",("number","5"),"*",("number","0"))) == ("number","0")
print optimize(("binop",("number","5"),"+",("number","0"))) == ("number","5")

