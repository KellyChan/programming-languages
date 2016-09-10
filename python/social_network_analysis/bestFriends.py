#            animal       speed   weight lifespan brain
#                          (mph)   (kg)  (years) mass (g)
animals = [("dog",          46,   35,     13,  280    ),
           ("elephant",     30, 3500,     50, 6250    ),
           ("frog",          5,    0.5,    8,    3    ),
           ("hippopotamus", 45, 1600,     45,  573    ),
           ("horse",        40,  385,     30, 642     ),
           ("human",        27,   80,     78, 2000    ),
           ("lion",         50,  250,     30,  454    ),
           ("mouse",         8,    0.025,  2,    0.625),
           ("rabbit",       25,    4,     12,   40    ), 
           ("shark",        26,  230,     20,   92    ),
           ("sparrow",      16,    0.024,  7,    2    )]
 
def importance_rank(items, weights):
    names = [item[0] for item in items]
    scores = [sum([a*b for (a,b) in zip(item[1:], weights)]) for item in items]
    results = zip(scores,names)
    res2 = sorted(results)
    return res2
 
answer = importance_rank(animals, (2,3,7,1))
 
for i in range(len(answer)):
    print i, answer[i][1], "(", answer[i][0], ")"