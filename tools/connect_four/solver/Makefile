CXX=g++
CXXFLAGS=--std=c++11 -W -Wall -O3 -DNDEBUG

SRCS=solver.cpp main.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: c4solver connect_four_solver connect_four_optimal_moves;

c4solver:$(OBJS)
	$(CXX) $(LDFLAGS) -o c4solver $(OBJS) $(LOADLIBES) $(LDLIBS)

connect_four_solver: solver.o connect_four_solver.o
	$(CXX) $(LDFLAGS) -o connect_four_solver solver.o connect_four_solver.o $(LDLIBS)

connect_four_optimal_moves: solver.o connect_four_optimal_moves.o
	$(CXX) $(LDFLAGS) -o connect_four_optimal_moves solver.o connect_four_optimal_moves.o $(LDLIBS)

.depend: $(SRCS)
	$(CXX) $(CXXFLAGS) -MM $^ > ./.depend
	
include .depend

clean:
	rm -f *.o .depend c4solver
