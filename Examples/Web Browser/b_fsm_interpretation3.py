# FSM Interpretation

# Provide s1 and s2 that are both accepted, but s1 != s2.

s1 = "bdgbdf"

s2 = "bdf"


edges = {(1,'a') : 2,
         (1,'b') : 3,
         (2,'c') : 4,
         (3,'d') : 5,
         (5,'c') : 2,
         (5,'f') : 6,
         (5,'g') : 1}

accepting = [6]


def fsmsim(string, current, edges, accepting):
    if string == "":
        return current in accepting
    else:
        letter = string[0]
        if (current, letter) in edges:
            destination = edges[(current, letter)]
            remaining_string = string[1:]
            return fsmsim(remaining_string, destination, edges, accepting)
        else:
            return False


print fsmsim(s1,1,edges,accepting)
# >>> True

print fsmsim(s2,1,edges,accepting)
# >>> True

print s1 != s2
# >>> True
