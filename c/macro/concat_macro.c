#include <stdio.h>

struct command
{
  char * name;
  void (*cmd)(void);
};


void quit_command(void)
{
  printf("quit_command\n");
}

void help_command(void)
{
  printf("help_command\n");
}

struct command commands[] = 
{
  { "quit", quit_command },
  { "help", help_command }
};

#define COMMAND(NAME) { #NAME, NAME ## _command }


struct command commands_new[] = 
{
  COMMAND(quit),
  COMMAND(help)
};


int main (void)
{
  printf("%s\n", commands[0].name);
  commands[0].cmd();

  int i;
  for (i = 0; i < 2; ++i)
  {
    printf("%s\n", commands_new[i].name);
    commands_new[i].cmd();
  }
}
