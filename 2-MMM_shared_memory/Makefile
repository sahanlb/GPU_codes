k40:mmm_shared.cu
	nvcc mmm_shared.cu -o mmm_shared -arch compute_35 -code sm_35

t102:mmm_shared.cu
	nvcc mmm_shared.cu -o mmm_shared -arch compute_75 -code sm_75

clean:
	rm -rf mmm_shared
