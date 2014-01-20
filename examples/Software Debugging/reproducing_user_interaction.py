# 
# Modify `input_command` so that it saves commands
# in `saved_commands`
# 

resume = True
command_index = 0
def original_input_command():
    global command_index

    commands = ["open", "save", "quit"]
    command = commands[command_index]
    command_index = command_index + 1
    return command

saved_commands = []
def input_command():
    command = original_input_command()
    # TODO: save the command in saved_commands
    global saved_commands
    saved_commands.append(command)
    return command

def process(command):
    global resume
    print repr(command)
    if command.startswith('q'):
        resume = False

def test():
    while resume:
        command = input_command()
        process(command)
    assert saved_commands == ['open', 'save', 'quit']

print repr(saved_commands)
