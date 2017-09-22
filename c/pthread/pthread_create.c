#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>


void * printer1 (void * ptr)
{
  char * thread = (char *)ptr;
  printf("%s: %d\n", thread, 3+3);
}


void * printer2 (void * ptr)
{
  char * thread = (char *)ptr;
  printf("%s: %d\n", thread, 4+6);
}


int main (void)
{
  pthread_t thread1, thread2;
  char * message1 = "Thread 1";
  char * message2 = "Thread 2";
  int iter1, iter2;

  iter1 = pthread_create (&thread1, NULL, printer1, (void*) message1);
  iter2 = pthread_create (&thread2, NULL, printer2, (void*) message2);

  pthread_join (thread1, NULL);
  pthread_join (thread2, NULL);

  printf ("Thread 1 returns: %d\n", iter1);
  printf ("Thread 2 returns: %d\n", iter2);

  exit(0);
}
