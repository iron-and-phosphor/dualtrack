all: main.cpp 
	clang++ main.cpp -o track -std=c++14 `pkg-config --cflags --libs opencv`