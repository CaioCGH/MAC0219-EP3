EXECS=mpi_hello_world mandelbrotCPU dmbrot
MPICC?=mpicc

all: ${EXECS}

dmbrot: dmbrot.c
	gcc -o dmbrot dmbrot.c -I .

mandelbrotCPU: mandelbrotCPU.c
	${MPICC} -o mandelbrotCPU mandelbrotCPU.c -lm -lpng -I .

mandelbrotGPU: mandelbrotGPU.cu
	nvcc -o mandelbrotGPU mandelbrotGPU.cu -lm -lpng -I .

clean:
	rm ${EXECS}