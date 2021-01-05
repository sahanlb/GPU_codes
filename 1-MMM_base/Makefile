k40:mmm_global.cu
	nvcc mmm_global.cu -o mmm_global -arch compute_35 -code sm_35

p100:mmm_global.cu
	nvcc mmm_global.cu -o mmm_global -arch compute_60 -code sm_60

t102:mmm_global.cu
	nvcc mmm_global.cu -o mmm_global -arch compute_75 -code sm_75

clean:
	rm -rf mmm_global
