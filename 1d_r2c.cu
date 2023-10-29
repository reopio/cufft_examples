#include <stdio.h>
#include <stdlib.h>
//#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <device_launch_parameters.h>


#define LENGTH 8
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

    float Data[LENGTH] = {2,1,-1,5,0,3,0,-4};
    cufftReal *CompData=(cufftReal*)malloc(LENGTH*sizeof(cufftReal));
    cufftComplex *CompData_C=(cufftComplex*)malloc(LENGTH*sizeof(cufftComplex));
    int i;
    for(i=0;i<LENGTH;i++)
    {
        CompData[i]=Data[i];
    }

    cufftReal *i_fftData,*i_fftData_out;
    cufftComplex *o_fftData;
    cudaMalloc((void**)&i_fftData,LENGTH*sizeof(cufftReal));
    cudaMalloc((void**)&i_fftData_out,LENGTH*sizeof(cufftReal));
    cudaMalloc((void**)&o_fftData,LENGTH*sizeof(cufftComplex));
    cudaMemcpy(i_fftData,CompData,LENGTH*sizeof(cufftReal),cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan,LENGTH,CUFFT_R2C,1);
    cufftExecR2C(plan,(cufftReal*)i_fftData,(cufftComplex*)o_fftData);
    cudaDeviceSynchronize();
    cudaMemcpy(CompData_C,o_fftData,LENGTH*sizeof(cufftComplex),cudaMemcpyDeviceToHost);


  printf("\nfft:\n");
  printf("--------------------\n");
  for(i=0;i<LENGTH;i++)
  {

    printf("%.3f",CompData_C[i].x);

    if(CompData_C[i].y != 0.0 )
    {
      printf("+%.3fi",CompData_C[i].y);
    } 
    printf("\n");
  }

  cudaMemset(CompData,0,LENGTH*sizeof(cufftReal));

  cufftPlan1d(&plan,LENGTH,CUFFT_C2R,1);
  cufftExecC2R(plan,(cufftComplex*)o_fftData,(cufftReal*)i_fftData_out);
  cudaDeviceSynchronize();
  cudaMemcpy(CompData,i_fftData_out,LENGTH*sizeof(cufftReal),cudaMemcpyDeviceToHost);

  printf("\n\ninverse fft:\n");
  printf("--------------------\n");
  for(i=0;i<LENGTH;i++)
  {
    printf("%.3f",CompData[i]/8.0);

    if(CompData[i] != 0 )
    {
      printf("+%.3fi",CompData[i]/8.0);
    } 
    printf("\n");
  }

     cufftDestroy(plan);
     free(CompData);
     free(CompData_C);
     cudaFree(o_fftData);
     cudaFree(i_fftData);
     cudaFree(i_fftData_out);

}