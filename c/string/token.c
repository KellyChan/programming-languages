#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char ** tokenize (const char * input, char * delim, int capacity)
{
  int count = 0;
  char * str = strdup(input);
  char ** result = malloc(capacity*sizeof(*result));

  char * tok = strtok(str, delim);
  while (1)
  {
    if (count >= capacity)
      result = realloc(result, (capacity *= 2)*sizeof(*result));
   
    result[count++] = tok ? strdup(tok) : tok;
    if (!tok) break;
    tok = strtok(NULL, delim);
  }

  free(str);
  return result;
}


int main()
{
  char ** tokens = tokenize ("1,w,n,c,w,n,c,h,w,2,3,4", ",", 10);
  char ** it;
  for (it = tokens; it && *it; ++it)
  {
    printf("%s\n", *it);
    free(*it);
  }

  free(tokens);
  return 0;
}
