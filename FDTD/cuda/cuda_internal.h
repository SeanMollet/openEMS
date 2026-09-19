/*
*	Copyright (C) 2026 Sean Mollet (sean@malmoset.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

// CUDA only: shared by the CUDA backend and the CUDA extensions

#ifndef CUDA_INTERNAL_H
#define CUDA_INTERNAL_H

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "gpu_backend_cuda.h"
#include "FDTD/gpu_coeff_sets.h"

//! Grid size (number of mesh lines) as passed to the kernels
struct CUDA_GridDim
{
	unsigned int nx, ny, nz;
};

//! Fields and coefficients use the ArrayNIJK layout of the host:
//! index(n,x,y,z) = n*nx*ny*nz + x*ny*nz + y*nz + z
__device__ __forceinline__ unsigned int nijk(const CUDA_GridDim& N, unsigned int n, unsigned int x, unsigned int y, unsigned int z)
{
	return ((n*N.nx + x)*N.ny + y)*N.nz + z;
}

//! Main update coefficients of node i, see GPU_Backend_CUDA::Impl::coeff_mode:
//! a[n*s] and b[n*s] are vv/vi (voltages, set_offset 0) or ii/iv (currents, set_offset 6)
//! of direction n. Mode 0: ca/cb are the full arrays, else ca is the table of sets.
struct CUDA_MainCoeff { const float* a; const float* b; unsigned int s; };

__device__ __forceinline__ CUDA_MainCoeff main_coeff(const void* index, const float* ca, const float* cb,
                                                     unsigned int mode, unsigned int sn, unsigned int i, unsigned int set_offset)
{
	CUDA_MainCoeff c;
	if (mode==0)
	{
		c.a = ca + i; c.b = cb + i; c.s = sn;
		return c;
	}
	const unsigned int set = (mode==1) ? (unsigned int)((const unsigned short*)index)[i] : ((const unsigned int*)index)[i];
	c.a = ca + 12*set + set_offset; c.b = c.a + 3; c.s = 1;
	return c;
}

#include "cuda_upml.cuh"

//! Throw a std::runtime_error if \a err is an error
void CUDA_Check(cudaError_t err, const char* what);

//! Device and work stream, shared by the backends of all grids of a simulation (see NewSubGridBackend())
struct CUDA_Context
{
	int device;
	cudaStream_t stream;
	std::string name;

	CUDA_Context();
	~CUDA_Context();
};

//! State of one grid
struct GPU_Backend_CUDA::Impl
{
	std::shared_ptr<CUDA_Context> ctx;

	CUDA_GridDim dim;
	size_t numCells;    //!< nx*ny*nz, the field buffers hold 3*numCells values

	float *volt, *curr;
	float2* energy;                      //!< per-line energy sums, see GPU_Backend_CUDA::CalcFastEnergy()
	size_t energy_count;
	std::vector<float2> energy_host;
	//! Main update coefficients: 0: full arrays vv, vi, ii, iv; 1/2: a 16/32 bit set index per node and the sets (see main_coeff())
	unsigned int coeff_mode;
	float *vv, *vi, *ii, *iv;
	void* index;
	float* coeff;

	//! The main updates cover the nodes in [main_start, main_stop), the fused UPML kernels the others (see cuda_ext_upml.cu)
	CUDA_GridDim main_start, main_stop;
	//! UPML extensions of this grid and whether they run fused with the main updates (-1: not decided yet), see cuda_ext_upml.cu
	std::vector<GPU_Extension*> upml;
	int upml_fused;
	//! UPML regions along z updated by the main kernels (nodes below main_start.nz / above main_stop.nz)
	CUDA_ZSlabs zslabs;
	bool zslab_lo, zslab_hi;

	//! Fused step (see update_fused): voltages and currents of the main nodes in one pass, from the
	//! current fields (volt, curr) into the next buffers (volt_next, curr_next), swapped afterwards.
	int fused_step;                          //!< -1: not decided yet, 0: off, 1: on
	int fused_blockers;                      //!< extensions or grids that do not allow it
	float *volt_next, *curr_next;
	std::vector<unsigned int> volt_modified; //!< flat indices of voltages changed between the half-steps (e.g. excitation)
	CUDA_FusedRegions fregions;              //!< the UPML regions in the kernel (see CUDA_Ext_UPML::CanFuse())
	unsigned int* fixup;                     //!< main nodes whose currents are recomputed after the voltage extensions
	unsigned int fixup_count;
	//! Decide once whether the fused step is used, and prepare it
	bool DecideFusedStep();

	Impl();
	~Impl();

	cudaStream_t Stream() const {return ctx->stream;}

	//! Device buffer of \a count elements, initialized with \a host or zero; freed with this Impl
	template <typename T>
	T* Alloc(size_t count, const T* host=NULL)
	{
		T* ptr = NULL;
		CUDA_Check(cudaMalloc(&ptr, std::max(count, (size_t)1)*sizeof(T)), "cudaMalloc");
		m_Allocations.push_back(ptr);
		// on the work stream: it does not synchronize with the legacy default stream
		if (host && count)
			CUDA_Check(cudaMemcpyAsync(ptr, host, count*sizeof(T), cudaMemcpyHostToDevice, Stream()), "cudaMemcpy");
		else
			CUDA_Check(cudaMemsetAsync(ptr, 0, std::max(count, (size_t)1)*sizeof(T), Stream()), "cudaMemset");
		Flush();
		return ptr;
	}

	//! Device buffer of the set indices of \a sets (16 or 32 bit)
	void* AllocIndex(const GPU_CoeffSets& sets)
	{
		if (sets.mode==1)
		{
			std::vector<unsigned short> index16(sets.index.begin(), sets.index.end());
			return Alloc<unsigned short>(index16.size(), index16.data());
		}
		return Alloc<unsigned int>(sets.index.size(), sets.index.data());
	}

	//! No UPML regions updated by the main kernels (call after changing main_start/main_stop)
	void ResetZSlabs();
	//! Nodes [B, E) of the main kernels: main_start/main_stop and the UPML regions along z they update
	void MainRange(CUDA_GridDim& B, CUDA_GridDim& E, unsigned int& threads_z);

	//! Wait until all work on the stream is done
	void Flush();

	//! Check for a kernel launch error
	void CheckLaunch(const char* kernel);

	//! Page-lock host memory (once) for fast transfers
	void PinHostMemory(void* ptr, size_t bytes);

	//! Launch geometry: one thread per (i,j,k), i fastest
	static dim3 Block(size_t ni, size_t nj);
	static dim3 Grid(dim3 block, size_t ni, size_t nj, size_t nk);

protected:
	std::vector<void*> m_Allocations;
	std::set<void*> m_Pinned;
};

//! Launch \a kernel with one thread per (i,j,k) on the stream of \a d; the kernel checks its bounds
//! Kernel timing for development (environment variable OPENEMS_CUDA_KERNEL_TIMES=1): the
//! time of every launch, per kernel name, printed at exit. Waits for each kernel.
struct CUDA_KernelTimes
{
	static bool Enabled();
	static void Begin(cudaStream_t stream);
	static void End(cudaStream_t stream, const char* name);
};

template <typename Kernel, typename... Args>
void CUDA_Launch(GPU_Backend_CUDA::Impl* d, const char* name, Kernel kernel, size_t ni, size_t nj, size_t nk, Args... args)
{
	if (ni==0 || nj==0 || nk==0)
		return;
	dim3 block = GPU_Backend_CUDA::Impl::Block(ni, nj);
	const bool timed = CUDA_KernelTimes::Enabled();
	if (timed)
		CUDA_KernelTimes::Begin(d->Stream());
	kernel<<<GPU_Backend_CUDA::Impl::Grid(block, ni, nj, nk), block, 0, d->Stream()>>>(args...);
	d->CheckLaunch(name);
	if (timed)
		CUDA_KernelTimes::End(d->Stream(), name);
}

//! Factory of a CUDA extension: the device implementation of \a eng_ext, or NULL if \a eng_ext is not of its type
typedef GPU_Extension* (*CUDA_ExtensionFactory)(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_Excitation(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_UPML(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_Mur_ABC(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_LorentzMaterial(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_LumpedRLC(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_TFSF(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_Absorbing_BC(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_SteadyState(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);
GPU_Extension* CUDA_CreateExt_Cylinder(GPU_Backend_CUDA::Impl* d, Engine_Extension* eng_ext, Engine* eng);

#endif // CUDA_INTERNAL_H
