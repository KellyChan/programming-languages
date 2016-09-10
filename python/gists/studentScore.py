lloyd = {
    "name": "Lloyd",
    "homework": [90.0, 97.0, 75.0, 92.0],
    "quizzes": [88.0, 40.0, 94.0],
    "tests": [75.0, 90.0]
}
alice = {
    "name": "Alice",
    "homework": [100.0, 92.0, 98.0, 100.0],
    "quizzes": [82.0, 83.0, 91.0],
    "tests": [89.0, 97.0]
}
tyler = {
    "name": "Tyler",
    "homework": [0.0, 87.0, 75.0, 22.0],
    "quizzes": [0.0, 75.0, 78.0],
    "tests": [100.0, 100.0]
}

# Add your function below!
def average(lst):
    return float(sum(lst)) / float(len(lst))

def get_average(student):
    homework = average(student['homework']) 
    quizzes = average(student['quizzes']) 
    tests = average(student['tests']) 
    return homework * .1 + quizzes * .3 + tests * .6


def get_letter_grade(score):
    if (score >= 90):
        return "A"
    elif (80 <= score < 90):
        return "B"
    elif (70 <= score < 80):
        return "C"
    elif (60 <= score < 70):
        return "D"
    else:
        return "F"

def get_class_average(cla):
    s = []
    for student in cla:
        s.append(get_average(student));
    return average(s)

score = get_average(lloyd)
print get_letter_grade(score)

students = [lloyd, alice, tyler]
print get_class_average(students)
print get_letter_grade(mean)
