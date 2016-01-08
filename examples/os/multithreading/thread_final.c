#define MAX 100 /* maximum number of images */ 

int bufavail = MAX; 
image_type frame_buf[MAX]; 
mutex_lock_type buflock;
cond_var_type buf_not_full; 
cond_var_type buf_not_empty; 

void digitizer() { 
  image_type dig_image; 
  int tail = 0; 

  while(1) { /* begin loop */
    grab(dig_image);

    thread_mutex_lock(buflock); 
    if (bufavail == 0) 
      thread_cond_wait(buf_not_full, 
           buflock); 
    thread_mutex_unlock(buflock);

    frame_buf[tail mod MAX] = dig_image; 
    tail = tail + 1; 

    thread_mutex_lock(buflock); 
    bufavail = bufavail - 1; 
    thread_cond_signal(buf_not_empty); 
    thread_mutex_unlock(buflock);
  }
}
void tracker() { 
  image_type track_image; 
  int head = 0; 
  while(1) { /* begin loop */ 
    thread_mutex_lock(buflock); 
    if (bufavail == MAX) 
      thread_cond_wait(buf_not_empty, 
           buflock); 
    thread_mutex_unlock(buflock); 

    track_image = frame_buf[head mod MAX]; 
    head = head + 1; 

    thread_mutex_lock(buflock); 
    bufavail = bufavail + 1; 
    thread_cond_signal(buf_not_full); 
    thread_mutex_unlock(buflock); 

    analyze(track_image); 
  } /* end loop */ 
} 

int main() 
{ 
  /* thread ids */ 
  thread_type digitizer_tid, tracker_tid; 

  /* create digitizer thread */ 
  digitizer_tid = thread_create(digitizer, NULL); 
  /* create tracker thread */ 
  tracker_tid = thread_create(tracker, NULL);

  /* rest of the code of main including 
   * termination conditions of the program 
   */ 
}
