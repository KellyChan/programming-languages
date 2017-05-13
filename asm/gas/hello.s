# ----------------------------------------------------------------------------------------
# run:
#     gcc -c hello.s && ld hello.o && ./a.out
#  or
#     gcc -nostdlib hello.s && ./a.out
# --------------------------------------------------------------------------------------- 

    .global _start
    .text

_start:
    # write (1, message, 13)
    mov $1, %rax                         # system call 1 is rax
    mov $1, %rdi                         # file handle 1 is stdout 
    mov $message, %rsi                   # address of string to output
    mov $13, %rdx                        # number of bytes
    syscall                              # invoke opearting system to do the write

    # exit(0)
    mov $60, %rax                        # system call 60 is exit
    xor %rdi, %rdi                       # we want reutrn code 0 
    syscall                              # invoke operating system to exit

message:
    .ascii  "Hello, world\n"
