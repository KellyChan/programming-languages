# FSM Simulation

edges = {(1, 'a') : 2,
         (2, 'a') : 2,
         (2, '1') : 3,
         (3, '1') : 3}

accepting = [3]

def fsmsim(string, current, edges, accepting):
    if string == "":
        return current in accepting
    else:
        letter = string[0]
        # QUIZ: You fill this out!
        # Is there a valid edge?
        # If so, take it.
        # If not, return False.
        # Hint: recursion.
        if (current, letter) in edges:
            destination = edges[(current,letter)]
            remaining_string = string[1:]
            return fsmsim(remaining_string,destination,edges,accepting)
        else:
            return False


print fsmsim("aaa111",1,edges,accepting)
# >>> True
