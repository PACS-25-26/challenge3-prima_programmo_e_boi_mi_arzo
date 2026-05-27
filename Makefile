CXX      ?= g++
CXXFLAGS ?= -std=c++20
CPPFLAGS ?= -fopenmp -O3 -Wall -I. -I/include -I/usr/include/eigen3 -I/usr/include/nlohmann
LDLIBS   ?= 
LINK.o := $(LINK.cc) # Use C++ linker.

DEPEND = make.dep

EXEC = main
SRCS = main.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY = all $(EXEC) $(OBJS) clean distclean doc $(DEPEND)

all: $(DEPEND) $(EXEC)

$(EXEC): $(OBJS)

$(OBJS): %.o: %.cpp

clean:
	$(RM) $(DEPEND)
	$(RM) *.o

distclean: clean
	$(RM) $(EXEC)
	$(RM) *.csv *.out *.bak *~
	$(RM) -r html/ latex/
	$(RM) Doxyfile

doc:
	doxygen -g Doxyfile && doxygen Doxyfile

$(DEPEND): $(SRCS)
	@$(RM) $(DEPEND)
	@for file in $(SRCS); \
	do \
	  $(CXX) $(CPPFLAGS) $(CXXFLAGS) -MM $${file} >> $(DEPEND); \
	done

-include $(DEPEND)
