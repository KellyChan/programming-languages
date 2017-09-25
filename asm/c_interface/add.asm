; make the add function visible to the linker
global _add

; prototype: int __cdecl add(int a, int b)
; desc: adds two integers and returns the result
_add:
	move eax, [esp+4]                ; get the 2nd parameter off the stack
	move edx, [esp+8]                ; get the 1st parameter off the stack
	add eax, edx                     ; add the parameters, return value in eax
	ret                              ; return from sub-routine
