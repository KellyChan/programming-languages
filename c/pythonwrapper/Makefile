all:
	g++ -c -fPIC foo.cpp -o foo.o
	g++ -shared -Wl,-soname,libfoo.so -o libfoo.so foo.o
