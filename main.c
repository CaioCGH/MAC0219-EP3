// #define DEBUG

#include <mpi.h>
#include <stdio.h>

#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <png.h>
#include <sys/time.h>
#include <string.h>

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

int writeImage(char* filename, int width, int height, float *buffer, char* title){
	int code = 0;
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
		png_charp key = (char *) "Title";
		title_text.key = key;
		title_text.text = title;
		png_set_text(png_ptr, info_ptr, &title_text, 1);
	}

	png_write_info(png_ptr, info_ptr);


	int x, y;
	/* é impossível paralelizar essa parte
	  Pela maneira como a libnpg foi implementada, a escrita
	  da imagem precisa ser sequancial, linha por linha.
	  Poderíamos até paralelizar o buffer todo de uma vez,
	  mas as funções de escrita na imagem seriam quadráticas
	  de qualquer maneira*/
	for (y = 0 ; y < height ; y++) {
		png_bytep row2 = (png_bytep) malloc(3 * width * sizeof(png_byte));
		for (x = 0 ; x < width ; x++) {
			setRGB(&(row2[x*3]), buffer[y*width + x]);
		}
		png_write_row(png_ptr, row2);
	}

	png_write_end(png_ptr, NULL);

	png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
	png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
	free(row);

	return code;
}

void createMandelbrotImage(float* buffer, int chunk_size, int y_0, int y_f, int width, int height, float a_0, float b_0, float a_1, float b_1, int maxIteration){
	int x, y;

	for (y = y_0; y < y_f; y++){
		//calculamos o valor de b do c para esta linha de  pixels
		float C_b = b_0 + (b_1-b_0)/ height*y;



		for (x = 0; x < width; x++){
			//calculamos o valor de a do c para esta coluna de pixels
			float C_a = a_0 + (a_1-a_0)/ (float) width*x;
			//começamos o valor de z_0 = a + bi = 0
			int j = 0;
			float a = 0;
			float b = 0;
			float z_mod = 0;
			buffer[(y - y_0) * width + x] = 0;

			//calculamos o valor de z_n:
			for(j = 0; j < maxIteration; j++){
				float a_tmp  = a*a - b*b + C_a;
				float b_tmp = 2*a*b + C_b;

				a = a_tmp;
				b = b_tmp;
				z_mod = sqrt(a*a + b*b);
				if (z_mod > 2){
					buffer[(y - y_0) * width + x] = j;
					break;
				}
			}
		}
	}
}
void scaleBufferSeq(float* buffer, int chunk_size, float min, float max){
	int i;
	for(i = 0; i < chunk_size; i++){
		buffer[i] = (buffer[i] - min) / (max - min);
	}
}

int main(int argc, char** argv) {
    float a_0 = atof(argv[1]);
	float b_0 = atof(argv[2]);
	float a_1 = atof(argv[3]);
	float b_1 = atof(argv[4]);
	int   width   = atoll(argv[5]);
	int   height  = atoll(argv[6]);
	// char* mode     = argv[7];
	int threads    = atoll(argv[8]);
	char* saida    = argv[9];
	int image_size = width*height;
	int maxIteration = 300;


    int world_size;
    int world_rank;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
	int i;
	for(i = 0; i < 13; i++)

    MPI_Init(&argc , &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Get_processor_name(processor_name, &name_len);
	printf("Creating mbrot in proc %d\n", world_rank);

	int chunk_height = height/world_size;
	int chunk_size = chunk_height*width;
	float* send_ch_buffer = (float*) malloc(chunk_size * sizeof(float));
	float* buffer = (float*) malloc(image_size * sizeof(float));

    call_me_maybe();

	MPI_Gather(send_ch_buffer, chunk_size, MPI_FLOAT, buffer, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);


	printf("%f\n", buffer[1]);
	// scaleBufferSeq(send_ch_buffer, chunk_size, 0, maxIteration);
	//
	//
	// if(world_rank == 0){
	// 	#ifdef DEBUG
	// 	for(i = 0; i < image_size; i++)
	// 		printf("buffer[%d] = %f\n",i,  buffer[i]);
	// 	#endif
	// 	writeImage(saida, width, height, buffer, (char *)"A Mandelbrot Image");
	// 	free(buffer);
	// }
	MPI_Finalize();
}
