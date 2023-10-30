all:
	nvcc 1d_c2c.cu -lcufft -o 1d_c2c
	nvcc 1d_r2c.cu -lcufft -o 1d_r2c
	nvcc 2d_c2c.cu -lcufft -o 2d_c2c
	nvcc 4d_c2c.cu -lcufft -o 4d_c2c
