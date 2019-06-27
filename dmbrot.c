#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main(int argc, char *argv[])
{
	float a_0   = atof(argv[1]);
	float b_0   = atof(argv[2]);
	float a_1   = atof(argv[3]);
	float b_1   = atof(argv[4]);
	int width   = atoll(argv[5]);
	int height  = atoll(argv[6]);
    char* mode  = argv[7];
	int threads = atoll(argv[8]);
	char* saida = argv[9];
    int i;

    int pid = fork();

	printf("mode: %s\n", mode);

    if (pid == 0){
        if(strcmp(mode, "cpu") == 0 || strcmp(mode, "CPU") == 0){
            char *argv2[13] = {"mpiexec", "-np", argv[8], "mandelbrotCPU", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]};
            char *cmd = (char *) malloc(1000 * sizeof(char));
            for(i = 0; i < 13; i++){
                strcat(cmd, argv2[i]);
                strcat(cmd, " ");
            }
            system(cmd);
            free(cmd);
        }
        if(strcmp(mode, "gpu") == 0 || strcmp(mode, "GPU") == 0){
			char *argv2[13] = {"mpiexec", "-np", argv[8], "./mandelbrotGPU", argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]};
            char *cmd = (char *) malloc(1000 * sizeof(char));
            for(i = 0; i < 13; i++){
                strcat(cmd, argv2[i]);
                strcat(cmd, " ");
            }
			system(cmd);
            free(cmd);
        }
    }
    else{
       waitpid(pid,0,0);
    }
    return 0;
}
