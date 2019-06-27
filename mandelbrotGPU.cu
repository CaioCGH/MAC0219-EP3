/*********************************************************************************************
*inspirado em http://www.labbookpages.co.uk/software/imgProc/files/libPNG/makePNG.c
*             https://computing.llnl.gov/tutorials/linux_clusters/gpu/NVIDIA.Introduction_to_CUDA_C.1.pdf
*
*********************************************************************************************/
// #define DEBUG
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <sys/time.h>
#include <string.h>
#include <cuda.h>
#include <mpi.h>


void setRGB(png_byte *ptr, float val){
	int v = (int)(val * 767);
	if (v < 0) v = 0;
	if (v > 767) v = 767;
	int offset = v % 256;

	if (v<256) {
		ptr[0] = 0; ptr[1] = 0; ptr[2] = offset;
	}
	else if (v<512) {
		ptr[0] = 0; ptr[1] = offset; ptr[2] = 255-offset;
	}
	else {
		ptr[0] = offset; ptr[1] = 255-offset; ptr[2] = 0;
	}
}
void writeImage(char* filename, int width, int height, float *buffer, char* title){
	FILE *fp = NULL;
	png_structp png_ptr = NULL;
	png_infop info_ptr = NULL;

	//Memória para uma linha de pixels, cada pixel usa 3 bytes (RGB)
	png_bytep row = (png_bytep) malloc(3 * width * sizeof(png_byte));

	fp = fopen(filename, "wb");

	png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height,
			8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
			PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);

	if (title != NULL) {
		png_text title_text;
		title_text.compression = PNG_TEXT_COMPRESSION_NONE;
		const char *name = "Title";
		title_text.key = (png_charp) name;
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
	}

	png_write_info(png_ptr, info_ptr);
	int x, y;
	for (y = 0 ; y < height; y++) {
		for (x = 0 ; x < width; x++) {
			setRGB(&(row[x*3]), buffer[y*width + x]);
		}
		png_write_row(png_ptr, row);
	}
	png_write_end(png_ptr, NULL);
	png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	free(row);
}

__global__
 void GPUimage(float *buffer, int width, int height, float a_0, float b_0, float a_1, float b_1, int maxIteration){

	int x;
	int y = threadIdx.x + blockIdx.x *blockDim.x;
	if(y < height){
		//calculamos o valor de b do c para esta linha de  pixels
		float C_b = b_0 + (b_1-b_0)/height*y;

		for (x = 0; x < width; x++){

			//calculamos o valor de a do c para esta coluna de pixels
			float C_a = a_0 + (a_1-a_0)/width*x;

			//começamos o valor de z_0 = a + bi = 0
			int j = 0;
			float a = 0;
			float b = 0;
			float z_mod = 0;


			//calculamos o valor de z_n:
			for(j = 0; j < maxIteration; j++){
				float a_tmp  = a*a - b*b + C_a;
				float b_tmp = 2*a*b + C_b;
				a = a_tmp;
				b = b_tmp;
				z_mod = sqrt(a*a + b*b);
				if (z_mod > 2){
					buffer[y * width + x] = j;
					break;
				}
			}
			if(j == maxIteration)
				buffer[y * width + x] = 0.0;
		}
	}
}

__global__
void scale_Buffer(float *buffer, int n, float min, float max){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	buffer[index] = (buffer[index] - min) / (max - min);
}

void get_Buffer_Extremes(float* buffer, int maxIteration, int n, int threads, float* min, float* max){
	*min = maxIteration;
	*max = 0;
	int i;
	for(i = 0; i < n; i++){
		if (buffer[i] > *max)
			*max = buffer[i];
		if (buffer[i] < *min)
			*min = buffer[i];
	}
}

void scaleBufferSeq(float* buffer, int n, float min, float max){
	int i;
	for(i = 0; i < n; i++){
		buffer[i] = (buffer[i] - min) / (max - min);
	}
}

int main(int argc, char *argv[])
{
	float a_0 = atof(argv[1]);
	float b_0 = atof(argv[2]);
	float a_1 = atof(argv[3]);
	float b_1 = atof(argv[4]);
	int   width   = atoll(argv[5]);
	int   height  = atoll(argv[6]);
	int maxInteration = 200;
	int threads   = atoll(argv[8]);
	char* saida    = argv[9];
	int n = width*height;
	cudaSetDevice(1);

    int world_size;
    int world_rank;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Get_processor_name(processor_name, &name_len);

	float *buffer;
	float *d_buffer;

	buffer = (float*) malloc(n * sizeof(float));
	d_buffer = (float*) malloc(n * sizeof(float));

	cudaMallocManaged(&d_buffer, n *sizeof(float));

	cudaMemcpy(d_buffer, buffer, n * sizeof(float), cudaMemcpyHostToDevice);


  	GPUimage<<<n/threads, threads>>>(d_buffer, width, height, a_0, b_0, a_1, b_1, maxInteration);
	cudaMemcpy(buffer, d_buffer, n * sizeof(float), cudaMemcpyDeviceToHost);

	float *min = (float*) malloc(sizeof(float));
	float *max = (float*) malloc(sizeof(float));
	get_Buffer_Extremes(buffer, maxInteration, n, threads, min, max);
	scale_Buffer<<<n/threads, threads>>>(d_buffer, n, *min, *max);

	cudaMemcpy(buffer, d_buffer, n * sizeof(float), cudaMemcpyDeviceToHost);

	#ifdef DEBUG
	printf("Printing buffer\n");
	int x, y;
	for (y = 0 ; y < height; y++) {
		for (x = 0 ; x < width; x++)
			printf("%f ", buffer[y*width + x]);
		printf("]\n");
	}
	#endif
	writeImage(saida, width, height, buffer, (char *) "A Mandelbrot Image");

	cudaFree(d_buffer);
	free(buffer);
    /*
	gettimeofday(&end, NULL);
	double elapsed_time = (end.tv_sec - start.tv_sec) +
                              (end.tv_usec - start.tv_usec) / 1000000.0;

	printf("%.4f\n", elapsed_time);
    */
	return 0;
}
