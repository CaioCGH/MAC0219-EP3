# MAC0219--EP3
```
make
```

Para rodar na CPU com OpenMP em 4 threads:

```
./dmbrot  0.3869 0.3099  0.3870 0.3100  1000 1000 cpu 4 imageCPU.png
```

Para rodar em GPU com Cuda com 32 threads por bloco:



```
./dmbrot  0.3869 0.3099  0.3870 0.3100  1000 1000 gpu 32 imageGPU.png
```

