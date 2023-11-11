#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cufft.h>

#define N 4
#define LEN 256

int main(){

    //float data[LEN];
    //for(unsigned i = 0;i<LEN;i++){
    //    data[i]=i+1;
    //}

    cufftComplex *data_h,*result_h;
    data_h=(cufftComplex*)malloc(LEN*sizeof(cufftComplex));
    result_h=(cufftComplex*)malloc(LEN*sizeof(cufftComplex));
    for(unsigned i = 0;i<LEN;i++){
        data_h[i].x=i+1;
        data_h[i].y=0;
    }

    //printf("data[%d]:%.8f\n",25,data_h[25].x);


    ///////////////// cufftPlanMany first 2 dims parameter ////////////
    cufftHandle plan1;
    //==========input===========//
    int n1[2] = {N,N};           // shape of each batch
    int rank1 = 2;               // dims of each batch
    int inembed1[N] = {N,N,N,N}; // main array dims
    int istride1 = 1;            // distance between two batches(row)
    int idist1 = N*N;            // distance between two batches(col)

    //==========output==========//
    int onembed1[N] = {N,N,N,N}; // main array dims
    int ostride1 = 1;            // distance between two batches(row)
    int odist1 = N*N;            // distance between two batches(col)
    int batch1 = N*N;            // nums of separated fft to parallel
    ///////////////// end of first 2 dims parameter ////////////

        ///////////////// cufftPlanMany rest 2 dims parameter ////////////
    cufftHandle plan2;
    //==========input===========//
    int n2[2] = {N,N};           // shape of each batch
    int rank2 = 2;               // dims of each batch
    int inembed2[N] = {N,N,N,N}; // main array dims
    int istride2 = N*N;          // distance between two batches(row)
    int idist2 = 1;              // distance between two batches(col)

    //==========output===========//
    int onembed2[N] = {N,N,N,N}; // main array dims
    int ostride2 = N*N;          // distance between two batches(row)
    int odist2 = 1;              // distance between two batches(col)
    int batch2 = N*N;            // nums of separated fft to parallel
    ///////////////// end of rest 2 dims parameter ////////////


    cufftComplex *data_g;
    cudaMalloc((void**)&data_g,LEN*sizeof(cufftComplex));
    cudaMemcpy(data_g,data_h,LEN*sizeof(cufftComplex),cudaMemcpyHostToDevice);

    cufftPlanMany(&plan1,rank1,n1,inembed1,istride1,idist1,onembed1,ostride1,odist1,CUFFT_C2C,batch1); // first fft of two dims of 4 dims
    cufftPlanMany(&plan2,rank2,n2,inembed2,istride2,idist2,onembed2,ostride2,odist2,CUFFT_C2C,batch2); // rest fft of two dims of 4 dims
    cufftExecC2C(plan1,data_g,data_g,CUFFT_FORWARD);
    cufftExecC2C(plan2,data_g,data_g,CUFFT_FORWARD);
    cudaDeviceSynchronize();

    //cudaMemcpy(result_h,data_g,LEN*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    cufftExecC2C(plan1,data_g,data_g,CUFFT_INVERSE);
    cufftExecC2C(plan2,data_g,data_g,CUFFT_INVERSE);
    cudaDeviceSynchronize();

    cudaMemcpy(result_h,data_g,LEN*sizeof(cufftComplex),cudaMemcpyDeviceToHost);

    int index;
    printf("check result(input index):\n");
    scanf("%d",&index);
    printf("data[%d]:%.8f\n",index,data_h[index].x);
    printf("result[%d]:%.8f\n",index,result_h[index].x/LEN);

    cufftDestroy(plan1);
    cufftDestroy(plan2);
    cudaFree(data_g);
    free(data_h);
    free(result_h);

    return 0;
}