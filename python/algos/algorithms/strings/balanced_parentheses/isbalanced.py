"""
Balanced braces question. ({[]}) Given a string of parantheses, brackets, and curly braces, write a function that returns whether the string is well balanced in that every left delimiter is closed by the correct right delimiter.
"""

def isbalanced(input_str):

    left_parens = {'{', '[', '('}
    right_parens = {'}', ']', ')'}
    right_to_left = {
                      '}': '{',
                      ']': '[',
                      ')': '('
                    }

    stack = []
    
    for char in input_str:
        if char in left_parens:
            stack.append(char)
        elif char in right_parens:
            if not stack or stack.pop() != right_to_left[char]:
                return False

    if stack:
        return False
    return True
