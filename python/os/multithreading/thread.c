#define MAX 100 /* maximum number of images */ 

int bufavail = MAX; 
image_type frame_buf[MAX]; 
mutex_lock_type buflock;

void digitizer() 
{ 
  image_type dig_image; 
  int tail = 0; 

  while(1) { /* begin loop */
    while (bufavail == 0); 

    grab(dig_image);
    frame_buf[tail mod MAX] = dig_image; 
    tail = tail + 1; 

    thread_mutex_lock(buflock); 
    bufavail = bufavail - 1;
    thread_mutex_unlock(buflock); 
  }
}

void tracker() { 
  image_type track_image; 
  int head = 0; 
  while(1) { /* begin loop */ 
    while (bufavail == MAX); 

    track_image = frame_buf[head mod MAX]; 
    head = head + 1; 

    thread_mutex_lock(buflock); 
    bufavail = bufavail + 1;
    thread_mutex_unlock(buflock); 

    analyze(track_image); 
  } 
  thread_mutex_unlock(buflock); 
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
