EXECS=mandelbrotCPU mandelbrotGPU dmbrot teste
MPICC?=mpicc

all: ${EXECS}

dmbrot: dmbrot.c
	gcc -o dmbrot dmbrot.c -I .

mandelbrotCPU: mandelbrotCPU.c
	${MPICC} -o mandelbrotCPU mandelbrotCPU.c -lm -lpng -I .

calc.o: calc.cu
	nvcc -c -lm -lpng -lmpi -I . calc.cu -o multiply.o

main: main.c
	mpicc -c -lm -lpng -lmpi -I . main.c


mandelbrotGPU: calc.o main.o
	mpicc -L/apps/CUDA/cuda-5.0/lib64/ -o mandelbrotGPU

teste: teste.c
	mpicc -o teste teste.c -lm -lpng -lmpi

clean:
	rm ${EXECS}
