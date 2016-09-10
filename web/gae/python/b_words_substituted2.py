# User Instructions
# 
# Write a function 'sub_m' that takes a 
# name and a nickname, and returns a 
# string of the following format: 
# "I'm NICKNAME. My real name is NAME, but my friends call me NICKNAME."
# 

given_string2 = "I'm %(nickname)s. My real name is %(name)s, but my friends call me %(nickname)s."
def sub_m(name, nickname):
    return given_string2 % {'name': name, 'nickname': nickname}
    
    

print sub_m("Mike", "Goose") 
# => "I'm Goose. My real name is Mike, but my friends call me Goose."
