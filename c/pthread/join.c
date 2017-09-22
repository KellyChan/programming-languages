#include <stdio.h>
#include <pthread.h>

#define NTHREADS 10

pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
int counter = 0;


void * thread_func (void * ptr)
{
  printf("Thread number %ld\n", pthread_self());
  pthread_mutex_lock (&mutex1);
  counter++;
  pthread_mutex_unlock (&mutex1);
}


int main ()
{
  pthread_t thread_id[NTHREADS];

  int i, j;

  for (i = 0; i < NTHREADS; i++)
  {
    pthread_create (&thread_id[i], NULL, thread_func, NULL);
  }

  for (j = 0; j < NTHREADS; j++)
  {
    pthread_join (thread_id[j], NULL);
  }

  printf("Final counter value: %d\n", counter);
}
