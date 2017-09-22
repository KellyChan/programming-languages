#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define COUNT_DONE 20
#define COUNT_HALT1 3
#define COUNT_HALT2 6

pthread_mutex_t count_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t condition_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t condition_cond = PTHREAD_COND_INITIALIZER;

int count = 0;

void * func1()
{
  for (;;)
  {
    pthread_mutex_lock (&condition_mutex);
    while (count >= COUNT_HALT1 && count <= COUNT_HALT2)
    {
      pthread_cond_wait (&condition_cond, &condition_mutex);
    }
    pthread_mutex_unlock (&condition_mutex);

    pthread_mutex_lock (&count_mutex);
    count++;
    printf("[Thread1] Counter value func1: %d\n", count);
    pthread_mutex_unlock (&count_mutex);

    if (count >= COUNT_DONE)  return (NULL);
  }
}


void * func2()
{
  for (;;)
  {
    pthread_mutex_lock (&condition_mutex);
    if (count < COUNT_HALT1 || count > COUNT_HALT2)
    {
      pthread_cond_signal (&condition_cond);
    }
    pthread_mutex_unlock (&condition_mutex);

    pthread_mutex_lock (&count_mutex);
    count++;
    printf("[Thread2] Counter value func2: %d\n", count);
    pthread_mutex_unlock (&count_mutex);

    if (count >= COUNT_DONE) return (NULL);
  }
}


int main()
{
  pthread_t thread1, thread2;

  pthread_create (&thread1, NULL, &func1, NULL);
  pthread_create (&thread2, NULL, &func2, NULL);
  pthread_join (thread1, NULL);
  pthread_join (thread2, NULL);

  exit(0);
}
