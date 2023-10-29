#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>

#define N 4
#define LENGTH 16
int main()
{

  int nDevices;
  //cudaDeviceProp
  cudaGetDeviceCount(&nDevices);
  
  printf("Number of devices: %d\n", nDevices);
  
  for (int i = 0; i < nDevices; i++) {
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (MHz): %d\n",
           prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }

    float Data[LENGTH] = {1,2,3,6,
    3,4,5,9,
    4,5,6,4,
    4,3,6,9};
    cufftComplex *CompData=(cufftComplex*)malloc(LENGTH*sizeof(cufftComplex));
    int i;
    for(i=0;i<LENGTH;i++)
    {
        CompData[i].x=Data[i];
        CompData[i].y=0;
    }

    cufftComplex *d_fftData;
    cudaMalloc((void**)&d_fftData,LENGTH*sizeof(cufftComplex));
    cudaMemcpy(d_fftData,CompData,LENGTH*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan2d(&plan,N,N,CUFFT_C2C);
    cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_FORWARD);
    cudaDeviceSynchronize();
    cudaMemcpy(CompData,d_fftData,LENGTH*sizeof(cufftComplex),cudaMemcpyDeviceToHost);


  printf("\nfft:\n");
  printf("--------------------\n");
  for(i=0;i<LENGTH;i++)
  {

    printf("%.3f",CompData[i].x);

    if(CompData[i].y != 0.0 )
    {
      printf("+%.3fi",CompData[i].y);
    } 
    if((i+1)%N==0){
      printf("\n");
    }else{
      printf(", ");
    }
  }

  cufftExecC2C(plan,(cufftComplex*)d_fftData,(cufftComplex*)d_fftData,CUFFT_INVERSE);
  cudaDeviceSynchronize();
  cudaMemcpy(CompData,d_fftData,LENGTH*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

  printf("\n\ninverse fft:\n");
  printf("--------------------\n");
  for(i=0;i<LENGTH;i++)
  {
    printf("%.3f",CompData[i].x/LENGTH);

    if(CompData[i].y != 0.0 )
    {
      printf("+%.3fi",CompData[i].y/LENGTH);
    } 
    if((i+1)%N==0){
      printf("\n");
    }else{
      printf(", ");
    }
  }
  printf("\n");

     cufftDestroy(plan);
     free(CompData);
     cudaFree(d_fftData);

}