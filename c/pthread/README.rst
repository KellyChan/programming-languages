#############################################################
pthread
#############################################################

- [pthread](http://www.cs.cmu.edu/afs/cs/academic/class/15492-f07/www/pthreads.html)


=============================================================
Thread Basic
=============================================================

- Thread operations include thread creation, termination, synchronization (joins,blocking), scheduling, data management and process interaction.
- A thread does not maintain a list of created threads, nor does it know the thread that created it.
- All threads within a process share the same address space.
- Threads in the same process share:
    - Process instructions
    - Most data
    - open files (descriptors)
    - signals and signal handlers
    - current working directory
    - User and group id
- Each thread has a unique:
    - Thread ID
    - set of registers, stack pointer
    - stack for local variables, return addresses
    - signal mask
    - priority
    - Return value: errno
- pthread functions return "0" if OK.

=============================================================
Thread Synchronization
=============================================================

The threads library provides three synchronization mechanisms:

- mutexes - Mutual exclusion lock: Block access to variables by other threads. This enforces exclusive access by a thread to a variable or set of variables.
- joins - Make a thread wait till others are complete (terminated).
- condition variables - data type pthread_cond_t
