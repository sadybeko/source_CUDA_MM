
CC = gcc
CXX = g++
NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -gencode=arch=compute_30,code=sm_30

CPPFLAGS = -Wall

.SUFFIXES: .cc .cu

.cc.o:
	$(CXX) $(CPPFLAGS) -O2 -c $<

.c.o:
	$(CC) $(CPPFLAGS) -O2 -c $<

.cu.o:
	$(NVCC) $(NVCC_FLAGS) -O2 -c $<

default: mm mm2
all: demo mm report.pdf
demo: demo-serial demo-parallel

demo-serial: demo-serial.o utils.o timer.o
	$(CXX) $(CPPFLAGS) $^ -o $@

demo-parallel: demo-parallel.o utils.o timer.o
	$(NVCC) $(NVCC_FLAGS) -O2 $^ -o $@

mm: mm.o utils.o timer.o
	$(NVCC) $(NVCC_FLAGS) -O2 $^ -o $@

mm2: mm2.o utils.o timer.o
	$(NVCC) $(NVCC_FLAGS) -O2 $^ -o $@

report.pdf: 
	pdflatex report.tex
	pdflatex report.tex
	pdflatex report.tex

clean:
	/bin/rm -f *.o *.aux *.log *.bbl *.blg *.toc
	/bin/rm -f demo-parallel demo-serial
	/bin/rm -f mm mm2

squeaky: clean
	/bin/rm -f *.synctex.gz *~ 
	/bin/rm -f report.pdf
