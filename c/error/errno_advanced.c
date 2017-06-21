/*
    errno value       Error
    1             / Operation not permitted /
    2             / No such file or directory /
    3             / No such process /
    4             / Interrupted system call /
    5             / I/O error /
    6             / No such device or address /
    7             / Argument list too long /
    8             / Exec format error /
    9             / Bad file number /
    10            / No child processes /
    11            / Try again /
    12            / Out of memory /
    13            / Permission denied /
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
 
int main ()
{
    FILE *fp;
 
    // If a file is opened which does not exist,
    // then it will be an error and corresponding
    // errno value will be set
    fp = fopen(" test.txt ", "r");
 
    // opening a file which does
    // not exist.
    printf("ErrorCode [%d]: %s\n ", errno, strerror(errno));
    perror("Message from perror");

 
    return 0;
}
