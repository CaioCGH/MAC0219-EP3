EXECS=mandelbrotCPU mandelbrotGPU dmbrot teste
MPICC?=mpicc
MPIFLAGS= -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -pthread -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi
all: ${EXECS}

dmbrot: dmbrot.c
	gcc -o dmbrot dmbrot.c -I .

mandelbrotCPU: mandelbrotCPU.c
	${MPICC} -o mandelbrotCPU mandelbrotCPU.c -lm -lpng -I .


mandelbrotGPU: mandelbrotGPU.cu
	nvcc ${MPIGLFAGS} -L/apps/CUDA/cuda-5.0/lib64/ -o mandelbrotGPU

teste: teste.c
	mpicc -o teste teste.c -lm -lpng -lmpi

clean:
	rm ${EXECS}
