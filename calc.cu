__global__
 void GPUimage(float *buffer, int width, int height, float a_0, float b_0, float a_1, float b_1, int maxIteration){

	int x;
	int y = threadIdx.x + blockIdx.x *blockDim.x;
	if(y < height){
	// for (y = 0; y < height; y++){

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
 extern "C" void call_me_maybe(){
    cudaSetDevice(1);

   float *buffer;
   float *d_buffer;

   buffer = (float*) malloc(n * sizeof(float));
   d_buffer = (float*) malloc(n * sizeof(float));

   cudaMallocManaged(&d_buffer, n *sizeof(float));
   cudaMemcpy(d_buffer, buffer, n * sizeof(float), cudaMemcpyHostToDevice);

   GPUimage<<<n/threads, threads>>>(d_buffer, width, height, a_0, b_0, a_1, b_1, maxInteration);
   cudaMemcpy(buffer, d_buffer, n * sizeof(float), cudaMemcpyDeviceToHost);
}
