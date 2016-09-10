"""
Project: Data Science 
Subject: Python - 3. Tennis

Author: Kelly Chan
Date: May 7 2014
"""

def createPlayers(n):
    A = ['a' + str(x+1) for x in range(n)]
    B = ['b' + str(x+1) for x in range(n)]
    return A, B

def createTeams(A, B):
    teams = []
    for i in range(len(A)):
        for j in range(len(B)):
            if i != j:
                teams.append((A[i], B[j]))
    return teams
    
def createSchedule(A, B):
    teams = []
    n = len(A)
    for i in range(n-1):
        for j in range(i+1, n):
            print "(%s, %s) vs. (%s, %s)" % (A[i], B[j], A[j], B[i])
            teams.append((A[i], B[j]))
            teams.append((A[j], B[i]))
    return teams


def main():

    n = 5
    A, B = createPlayers(n) 
    teams = createTeams(A, B) 
    teams_check = createSchedule(A, B)

    assert len(teams) == len(teams_check)
    assert set(teams) == set(teams_check)
    

if __name__ == '__main__':
    main()
