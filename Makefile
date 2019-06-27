EXECS=mandelbrotCPU mandelbrotGPU dmbrot teste
MPICC?=mpicc
MPIFLAGS=-I/usr/local/include -L/usr/local/lib -lmpi
all: ${EXECS}

dmbrot: dmbrot.c
	gcc -o dmbrot dmbrot.c -I .

mandelbrotCPU: mandelbrotCPU.c
	${MPICC} -o mandelbrotCPU mandelbrotCPU.c -lm -lpng -I . 


mandelbrotGPU: mandelbrotGPU.cu
	nvcc ${MPIFLAGS} -L/apps/CUDA/cuda-5.0/lib64/ -o mandelbrotGPU mandelbrotGPU.cu -lm -lpng 

teste: teste.c
	mpicc -o teste teste.c -lm -lpng -lmpi

clean:
	rm ${EXECS}
