// EMD approximation module (based on auction algorithm)
// author: Renwu Li
// torch version: https://github.com/Colin97/MSN-Point-Cloud-Completion/blob/HEAD/emd/
#include <stdio.h>

#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__global__ void clear(int b, int * cnt_tmp, int * unass_cnt) {
	for (int i = threadIdx.x; i < b; i += blockDim.x) {
		cnt_tmp[i] = 0;
		unass_cnt[i] = 0;
	}
}

__global__ void calc_unass_cnt(int b, int n, int * assignment, int * unass_cnt) { 
	// count the number of unassigned points in each batch
	const int BLOCK_SIZE = 1024; 
	__shared__ int scan_array[BLOCK_SIZE];
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		scan_array[threadIdx.x] = assignment[i * n + blockIdx.y * BLOCK_SIZE + threadIdx.x] == -1 ? 1 : 0;
		__syncthreads();
		
		int stride = 1;
		while(stride <= BLOCK_SIZE / 2) {
			int index = (threadIdx.x + 1) * stride * 2 - 1; 
			if(index < BLOCK_SIZE)
				scan_array[index] += scan_array[index - stride]; 
			stride = stride * 2;
			__syncthreads(); 
		}
		__syncthreads();
		
		if (threadIdx.x == BLOCK_SIZE - 1) {
			atomicAdd(&unass_cnt[i], scan_array[threadIdx.x]);
		}
		__syncthreads();
	}
}

__global__ void calc_unass_cnt_sum(int b, int * unass_cnt, int * unass_cnt_sum) {
	// count the cumulative sum over over unass_cnt
	const int BLOCK_SIZE = 512; // batch_size <= 512
	__shared__ int scan_array[BLOCK_SIZE];
	scan_array[threadIdx.x] = unass_cnt[threadIdx.x];
	__syncthreads();
	
	int stride = 1;
	while(stride <= BLOCK_SIZE / 2) {
		int index = (threadIdx.x + 1) * stride * 2 - 1; 
		if(index < BLOCK_SIZE)
			scan_array[index] += scan_array[index - stride]; 
		stride = stride * 2;
		__syncthreads(); 
	}
	__syncthreads();
	stride = BLOCK_SIZE / 4; 
	while(stride > 0) {
		int index = (threadIdx.x + 1) * stride * 2 - 1; 
		if((index + stride) < BLOCK_SIZE)
			scan_array[index + stride] += scan_array[index];
		stride = stride / 2;
		__syncthreads(); 
	}
	__syncthreads(); 
	
	//printf("%d\n", unass_cnt_sum[b - 1]);
	unass_cnt_sum[threadIdx.x] = scan_array[threadIdx.x];
}

__global__ void calc_unass_idx(int b, int n, int * assignment, int * unass_idx, int * unass_cnt, int * unass_cnt_sum, int * cnt_tmp) {
	// list all the unassigned points
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		if (assignment[i * n + blockIdx.y * 1024 + threadIdx.x] == -1) {
			int idx = atomicAdd(&cnt_tmp[i], 1);
			unass_idx[unass_cnt_sum[i] - unass_cnt[i] + idx] = blockIdx.y * 1024 + threadIdx.x;
		} 
	}
}

__global__ void Bid(int b, int n, const float * xyz1, const float * xyz2, float eps, int * assignment, int * assignment_inv, float * price, 
					int * bid, float * bid_increments, float * max_increments, int * unass_cnt, int * unass_cnt_sum, int * unass_idx) {
	const int batch = 2048, block_size = 1024, block_cnt = n / 1024;
	__shared__ float xyz2_buf[batch * 3];
	__shared__ float price_buf[batch];
	__shared__ float best_buf[block_size];
	__shared__ float better_buf[block_size];
	__shared__ int best_i_buf[block_size];
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		int _unass_cnt = unass_cnt[i];
		if (_unass_cnt == 0)
			continue;
		int _unass_cnt_sum = unass_cnt_sum[i];
		int unass_per_block = (_unass_cnt + block_cnt - 1) / block_cnt;
		int thread_per_unass = block_size / unass_per_block;
		int unass_this_block = max(min(_unass_cnt - (int) blockIdx.y * unass_per_block, unass_per_block), 0);
			
		float x1, y1, z1, best = -1e9, better = -1e9;
		int best_i = -1, _unass_id = -1, thread_in_unass;

		if (threadIdx.x < thread_per_unass * unass_this_block) {
			_unass_id = unass_per_block * blockIdx.y + threadIdx.x / thread_per_unass + _unass_cnt_sum - _unass_cnt;
			_unass_id = unass_idx[_unass_id];
			thread_in_unass = threadIdx.x % thread_per_unass;

			x1 = xyz1[(i * n + _unass_id) * 3 + 0];
			y1 = xyz1[(i * n + _unass_id) * 3 + 1];
			z1 = xyz1[(i * n + _unass_id) * 3 + 2];
		}

		for (int k2 = 0; k2 < n; k2 += batch) {
			int end_k = min(n, k2 + batch) - k2;
			for (int j = threadIdx.x; j < end_k * 3; j += blockDim.x) {
				xyz2_buf[j] = xyz2[(i * n + k2) * 3 + j];
			}
			for (int j = threadIdx.x; j < end_k; j += blockDim.x) {
				price_buf[j] = price[i * n + k2 + j];
			}
			__syncthreads();

			if (_unass_id != -1) {
				int delta = (end_k + thread_per_unass - 1) / thread_per_unass;
				int l = thread_in_unass * delta;
				int r = min((thread_in_unass + 1) * delta, end_k);
				for (int k = l; k < r; k++) 
				//if (!last || assignment_inv[i * n + k + k2] == -1)
				{
					float x2 = xyz2_buf[k * 3 + 0] - x1;
					float y2 = xyz2_buf[k * 3 + 1] - y1;
					float z2 = xyz2_buf[k * 3 + 2] - z1;
					// the coordinates of points should be normalized to [0, 1]
					float d = 3.0 - sqrtf(x2 * x2 + y2 * y2 + z2 * z2) - price_buf[k];
					if (d > best) {
						better = best;
						best = d;
						best_i = k + k2;
					}
					else if (d > better) {
						better = d;
					}
				}
			}
			__syncthreads();
		}

		best_buf[threadIdx.x] = best;
		better_buf[threadIdx.x] = better;
		best_i_buf[threadIdx.x] = best_i;
		__syncthreads();
		
		if (_unass_id != -1 && thread_in_unass == 0) {
			for (int j = threadIdx.x + 1; j < threadIdx.x + thread_per_unass; j++) {
				if (best_buf[j] > best) {
					better = max(best, better_buf[j]);
					best = best_buf[j];
					best_i = best_i_buf[j];
				}
				else better = max(better, best_buf[j]);
			}
			bid[i * n + _unass_id] = best_i;
			bid_increments[i * n + _unass_id] = best - better + eps; 
			atomicMax(&max_increments[i * n + best_i], best - better + eps);
		}
	}
}

__global__ void GetMax(int b, int n, int * assignment, int * bid, float * bid_increments, float * max_increments, int * max_idx) {
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		int j = threadIdx.x + blockIdx.y * blockDim.x;
		if (assignment[i * n + j] == -1) {
			int bid_id = bid[i * n + j];
			float bid_inc = bid_increments[i * n + j];
			float max_inc = max_increments[i * n + bid_id];
			if (bid_inc - 1e-6 <= max_inc && max_inc <= bid_inc + 1e-6) 
			{
				max_idx[i * n + bid_id] = j;
			}
		}
	}
}

__global__ void Assign(int b, int n, int * assignment, int * assignment_inv, float * price, int * bid, float * bid_increments, float * max_increments, int * max_idx, bool last) {
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		int j = threadIdx.x + blockIdx.y * blockDim.x;
		if (assignment[i * n + j] == -1) {
			int bid_id = bid[i * n + j];
			if (last || max_idx[i * n + bid_id] == j) 
			{
				float bid_inc = bid_increments[i * n + j];
				int ass_inv = assignment_inv[i * n + bid_id];
				if (!last && ass_inv != -1) {
					assignment[i * n + ass_inv] = -1;
				}
				assignment_inv[i * n + bid_id] = j;
				assignment[i * n + j] = bid_id;
				price[i * n + bid_id] += bid_inc;
				max_increments[i * n + bid_id] = -1e9;
			}
		}
	}
}

__global__ void CalcDist(int b, int n, const float * xyz1, const float * xyz2, float * dist, const int * assignment) {
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		int j = threadIdx.x + blockIdx.y * blockDim.x;
		int k = assignment[i * n + j];
		float deltax = xyz1[(i * n + j) * 3 + 0] - xyz2[(i * n + k) * 3 + 0];
		float deltay = xyz1[(i * n + j) * 3 + 1] - xyz2[(i * n + k) * 3 + 1];
		float deltaz = xyz1[(i * n + j) * 3 + 2] - xyz2[(i * n + k) * 3 + 2];
		dist[i * n + j] = deltax * deltax + deltay * deltay + deltaz * deltaz;
	}
}


int emdMatchLauncher(int b, int n, int m, \
                    const float* xyz1, \
                    const float* xyz2, \
                    int* assignment, \
                    float* price, \
                    int* assignment_inv, \
                    int* bid, \
                    float* bid_increments, \
                    float* max_increments, \
                    int* unass_idx, \
                    int* unass_cnt, \
                    int* unass_cnt_sum, \
                    int* cnt_tmp, \
                    int* max_idx, \
                    float eps,
                    int iters
) {

    if (n != m) {
        printf("Input error! The two point clouds should have the same size!\n");
        return -1;
    }

    if (b > 512) {
        printf("Input error! The batch size should not exceed 512!\n");
        return -1;
    }

    if (n % 1024 != 0) {
        printf("Input error! The size of the input point clouds should be a multiple of 1024.\n");
        return -1;
    }

	cudaMemset(assignment, -1, b * n * sizeof(int));
	cudaMemset(assignment_inv, -1, b * n * sizeof(int));

    for (int i = 0; i < iters; i++) {
		clear<<<1, b>>>(b, cnt_tmp, unass_cnt);
		calc_unass_cnt<<<dim3(b, n / 1024, 1), 1024>>>(b, n, assignment, unass_cnt);
		calc_unass_cnt_sum<<<1, b>>>(b, unass_cnt, unass_cnt_sum);
		calc_unass_idx<<<dim3(b, n / 1024, 1), 1024>>>(b, n, assignment, unass_idx, unass_cnt, 
											 unass_cnt_sum, cnt_tmp);
		Bid<<<dim3(b, n / 1024, 1), 1024>>>(b, n, xyz1, xyz2, eps, assignment, assignment_inv, 
			                          price, bid, bid_increments, max_increments,
			                          unass_cnt, unass_cnt_sum, unass_idx);
		GetMax<<<dim3(b, n / 1024, 1), 1024>>>(b, n, assignment, bid, bid_increments, max_increments, max_idx);
		Assign<<<dim3(b, n / 1024, 1), 1024>>>(b, n, assignment, assignment_inv, price, bid,
									  bid_increments, max_increments, max_idx, i == iters - 1);

	}
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error in nnd Output: %s\n", cudaGetErrorString(err));
		return 0;
	}
	return 1;
}


int emdCostLauncher(int b, int n, const float* xyz1, const float* xyz2, float* dist, const int* assignment) {
	CalcDist<<<dim3(b, n / 1024, 1), 1024>>>(b, n, xyz1, xyz2, dist, assignment);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error in nnd Output: %s\n", cudaGetErrorString(err));
		return 0;
	}
	return 1;
}

__global__ void emdcostGrad(int b, int n, int m, const float* xyz1, const float* xyz2, const int* assignment, float* grad_xyz) {    
	for (int i = blockIdx.x; i < b; i += gridDim.x) {
		for (int j = threadIdx.x + blockIdx.y * blockDim.x; j < n; j += blockDim.x * gridDim.y) {
			float x1 = xyz1[(i * n + j) * 3 + 0];
			float y1 = xyz1[(i * n + j) * 3 + 1];
			float z1 = xyz1[(i * n + j) * 3 + 2];
			int j2 = assignment[i * n + j];
			float x2 = xyz2[(i * n + j2) * 3 + 0];
			float y2 = xyz2[(i * n + j2) * 3 + 1];
			float z2 = xyz2[(i * n + j2) * 3 + 2];
			atomicAdd(&(grad_xyz[(i * n + j) * 3 + 0]), x1 - x2);
			atomicAdd(&(grad_xyz[(i * n + j) * 3 + 1]), y1 - y2);
			atomicAdd(&(grad_xyz[(i * n + j) * 3 + 2]), z1 - z2);
		}
	}
}

void emdcostGradLauncher(int b, int n, int m, const float* xyz1, const float* xyz2, const int* assignment, float* grad_xyz) {	
	emdcostGrad<<<dim3(b, n / 1024, 1), 1024>>>(b, n, m, xyz1, xyz2, assignment, grad_xyz);
}
