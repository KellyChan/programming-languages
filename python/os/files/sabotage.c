#include<stdlib.h>
#include<unistd.h>
#include <fcntl.h>
#include<stdio.h>


int main(int argc, char **argv){
int fd;
ssize_t len;
char *filename;
int key, srch_key, new_value;

if(argc < 4){
  fprintf(stderr, "usage: sabotage filename key value\n");
  exit(EXIT_FAILURE);
}

filename = argv[1];
srch_key = strtol(argv[2], NULL, 10);
new_value = strtol(argv[3], NULL, 10);

fd = open(filename, O_RDWR);

while( sizeof(int) == read(fd, &key, sizeof(int)) ){
  if (key != srch_key)
    lseek(fd, sizeof(int), SEEK_CUR);
  else{
    write(fd, &new_value, sizeof(int));
    close(fd);
    return EXIT_SUCCESS;
  }
}

fprintf(stderr, "Key not found!");
return EXIT_FAILURE;
}
