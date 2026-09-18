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

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "cuda_internal.h"
#include "FDTD/operator.h"

// Main FDTD updates, operation by operation the same as Engine::UpdateVoltages/UpdateCurrents.
// One thread per mesh node: z along x-threads (contiguous), y along y-threads, x along the grid z-dimension.
__global__ void update_voltages(float* __restrict__ volt, const float* __restrict__ curr,
                                const float* __restrict__ vv, const float* __restrict__ vi, CUDA_GridDim N)
{
	const unsigned int z = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = blockIdx.z;
	if (z>=N.nz || y>=N.ny || x>=N.nx)
		return;
	const unsigned int sn = N.nx*N.ny*N.nz;
	const unsigned int i  = nijk(N, 0, x, y, z);
	const unsigned int xm = (x>0) ? N.ny*N.nz : 0;   // shift to the previous line, none on the first
	const unsigned int ym = (y>0) ? N.nz : 0;
	const unsigned int zm = (z>0) ? 1 : 0;
	float v;

	//for x
	v  = volt[i] * vv[i];
	v += vi[i] * (curr[2*sn+i] - curr[2*sn+i-ym] - curr[sn+i] + curr[sn+i-zm]);
	volt[i] = v;

	//for y
	v  = volt[sn+i] * vv[sn+i];
	v += vi[sn+i] * (curr[i] - curr[i-zm] - curr[2*sn+i] + curr[2*sn+i-xm]);
	volt[sn+i] = v;

	//for z
	v  = volt[2*sn+i] * vv[2*sn+i];
	v += vi[2*sn+i] * (curr[sn+i] - curr[sn+i-xm] - curr[i] + curr[i-ym]);
	volt[2*sn+i] = v;
}

// the currents on the last mesh line of each direction are not updated
__global__ void update_currents(float* __restrict__ curr, const float* __restrict__ volt,
                                const float* __restrict__ ii, const float* __restrict__ iv, CUDA_GridDim N)
{
	const unsigned int z = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	const unsigned int x = blockIdx.z;
	if (z+1>=N.nz || y+1>=N.ny || x+1>=N.nx)
		return;
	const unsigned int sn = N.nx*N.ny*N.nz;
	const unsigned int i  = nijk(N, 0, x, y, z);
	const unsigned int xp = N.ny*N.nz;
	const unsigned int yp = N.nz;
	float c;

	//for x
	c  = curr[i] * ii[i];
	c += iv[i] * (volt[2*sn+i] - volt[2*sn+i+yp] - volt[sn+i] + volt[sn+i+1]);
	curr[i] = c;

	//for y
	c  = curr[sn+i] * ii[sn+i];
	c += iv[sn+i] * (volt[i] - volt[i+1] - volt[2*sn+i] + volt[2*sn+i+xp]);
	curr[sn+i] = c;

	//for z
	c  = curr[2*sn+i] * ii[2*sn+i];
	c += iv[2*sn+i] * (volt[sn+i] - volt[sn+i+xp] - volt[i] + volt[i+yp]);
	curr[2*sn+i] = c;
}

/***************************** helpers *****************************/

void CUDA_Check(cudaError_t err, const char* what)
{
	if (err!=cudaSuccess)
		throw std::runtime_error(std::string("GPU_Backend_CUDA: ") + what + " failed: " + cudaGetErrorString(err));
}

CUDA_Context::CUDA_Context()
{
	device = 0;
	stream = 0;
}

CUDA_Context::~CUDA_Context()
{
	if (stream)
	{
		cudaStreamSynchronize(stream);
		cudaStreamDestroy(stream);
	}
}

GPU_Backend_CUDA::Impl::Impl()
{
	dim.nx = dim.ny = dim.nz = 0;
	numCells = 0;
	volt = curr = vv = vi = ii = iv = NULL;
}

GPU_Backend_CUDA::Impl::~Impl()
{
	if (ctx)
		cudaStreamSynchronize(ctx->stream);
	for (size_t n=0; n<m_Allocations.size(); ++n)
		cudaFree(m_Allocations.at(n));
	for (std::set<void*>::iterator it=m_Pinned.begin(); it!=m_Pinned.end(); ++it)
		cudaHostUnregister(*it);
}

void GPU_Backend_CUDA::Impl::Flush()
{
	CUDA_Check(cudaStreamSynchronize(Stream()), "stream synchronization");
}

void GPU_Backend_CUDA::Impl::CheckLaunch(const char* kernel)
{
	CUDA_Check(cudaGetLastError(), kernel);
}

void GPU_Backend_CUDA::Impl::PinHostMemory(void* ptr, size_t bytes)
{
	if (m_Pinned.count(ptr))
		return;
	// not fatal: unpinned memory only makes the transfers slower
	if (cudaHostRegister(ptr, bytes, cudaHostRegisterDefault)==cudaSuccess)
		m_Pinned.insert(ptr);
	else
		cudaGetLastError();
}

dim3 GPU_Backend_CUDA::Impl::Block(size_t ni, size_t nj)
{
	if (nj<=1)
		return dim3(256, 1, 1);
	if (ni<=16)
		return dim3(16, 16, 1);
	return dim3(32, 8, 1);
}

dim3 GPU_Backend_CUDA::Impl::Grid(dim3 block, size_t ni, size_t nj, size_t nk)
{
	return dim3((ni+block.x-1)/block.x, (nj+block.y-1)/block.y, nk);
}

/***************************** GPU_Backend_CUDA *****************************/

GPU_Backend_CUDA* GPU_Backend_CUDA::New()
{
	int count = 0;
	if ((cudaGetDeviceCount(&count)!=cudaSuccess) || (count==0))
	{
		cudaGetLastError();
		return NULL;
	}
	Impl* impl = new Impl();
	impl->ctx = std::make_shared<CUDA_Context>();
	CUDA_Check(cudaGetDevice(&impl->ctx->device), "cudaGetDevice");
	cudaDeviceProp prop;
	CUDA_Check(cudaGetDeviceProperties(&prop, impl->ctx->device), "cudaGetDeviceProperties");
	impl->ctx->name = prop.name;
	CUDA_Check(cudaStreamCreateWithFlags(&impl->ctx->stream, cudaStreamNonBlocking), "cudaStreamCreate");
	return new GPU_Backend_CUDA(impl);
}

GPU_Backend* GPU_Backend_CUDA::NewSubGridBackend()
{
	Impl* impl = new Impl();
	impl->ctx = d->ctx;
	return new GPU_Backend_CUDA(impl);
}

GPU_Backend_CUDA::GPU_Backend_CUDA(Impl* impl)
{
	d = impl;
}

GPU_Backend_CUDA::~GPU_Backend_CUDA()
{
	delete d;
}

std::string GPU_Backend_CUDA::GetName() const
{
	return "CUDA (" + d->ctx->name + ")";
}

bool GPU_Backend_CUDA::Init(const Operator* op)
{
	unsigned int numLines[3];
	for (int n=0; n<3; ++n)
		numLines[n] = op->GetNumberOfLines(n, true);
	d->dim.nx = numLines[0];
	d->dim.ny = numLines[1];
	d->dim.nz = numLines[2];
	d->numCells = (size_t)numLines[0]*numLines[1]*numLines[2];

	// the kernels index with 32 bit
	if (3*d->numCells > std::numeric_limits<unsigned int>::max())
	{
		std::cerr << "GPU_Backend_CUDA::Init: Error: the mesh is too large for 32 bit indexing" << std::endl;
		return false;
	}

	const size_t count = 3*d->numCells;
	d->volt = d->Alloc<float>(count);
	d->curr = d->Alloc<float>(count);

	// upload the final operator coefficients, including all changes by operator extensions
	std::vector<float> vv(count), vi(count), ii(count), iv(count);
	unsigned int pos[3];
	size_t idx = 0;
	for (int n=0; n<3; ++n)
		for (pos[0]=0; pos[0]<numLines[0]; ++pos[0])
			for (pos[1]=0; pos[1]<numLines[1]; ++pos[1])
				for (pos[2]=0; pos[2]<numLines[2]; ++pos[2], ++idx)
				{
					vv[idx] = op->GetVV(n, pos[0], pos[1], pos[2]);
					vi[idx] = op->GetVI(n, pos[0], pos[1], pos[2]);
					ii[idx] = op->GetII(n, pos[0], pos[1], pos[2]);
					iv[idx] = op->GetIV(n, pos[0], pos[1], pos[2]);
				}
	d->vv = d->Alloc<float>(count, vv.data());
	d->vi = d->Alloc<float>(count, vi.data());
	d->ii = d->Alloc<float>(count, ii.data());
	d->iv = d->Alloc<float>(count, iv.data());
	return true;
}

void GPU_Backend_CUDA::UpdateVoltages()
{
	CUDA_Launch(d, "update_voltages", update_voltages, d->dim.nz, d->dim.ny, d->dim.nx,
	            d->volt, (const float*)d->curr, (const float*)d->vv, (const float*)d->vi, d->dim);
}

void GPU_Backend_CUDA::UpdateCurrents()
{
	CUDA_Launch(d, "update_currents", update_currents, d->dim.nz, d->dim.ny, d->dim.nx,
	            d->curr, (const float*)d->volt, (const float*)d->ii, (const float*)d->iv, d->dim);
}

static void CheckSize(size_t host_count, size_t device_count)
{
	if (host_count!=device_count)
		throw std::runtime_error("GPU_Backend_CUDA: host and device field size mismatch");
}

void GPU_Backend_CUDA::DownloadVoltages(ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory(volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	CUDA_Check(cudaMemcpyAsync(volt.data(), d->volt, volt.size()*sizeof(float), cudaMemcpyDeviceToHost, d->Stream()), "download");
	d->Flush();
}

void GPU_Backend_CUDA::DownloadCurrents(ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory(curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	CUDA_Check(cudaMemcpyAsync(curr.data(), d->curr, curr.size()*sizeof(float), cudaMemcpyDeviceToHost, d->Stream()), "download");
	d->Flush();
}

void GPU_Backend_CUDA::UploadVoltages(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt)
{
	CheckSize(volt.size(), 3*d->numCells);
	d->PinHostMemory(volt.data(), volt.size()*sizeof(FDTD_FLOAT));
	CUDA_Check(cudaMemcpyAsync(d->volt, volt.data(), volt.size()*sizeof(float), cudaMemcpyHostToDevice, d->Stream()), "upload");
	d->Flush();
}

void GPU_Backend_CUDA::UploadCurrents(const ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr)
{
	CheckSize(curr.size(), 3*d->numCells);
	d->PinHostMemory(curr.data(), curr.size()*sizeof(FDTD_FLOAT));
	CUDA_Check(cudaMemcpyAsync(d->curr, curr.data(), curr.size()*sizeof(float), cudaMemcpyHostToDevice, d->Stream()), "upload");
	d->Flush();
}

void GPU_Backend_CUDA::Synchronize()
{
	d->Flush();
}

// all CUDA extensions, see cuda_internal.h
static const CUDA_ExtensionFactory CUDA_EXTENSIONS[] = {
	CUDA_CreateExt_Excitation,
	CUDA_CreateExt_UPML,
};

GPU_Extension* GPU_Backend_CUDA::CreateExtension(Engine_Extension* eng_ext, Engine* eng)
{
	for (size_t n=0; n<sizeof(CUDA_EXTENSIONS)/sizeof(CUDA_EXTENSIONS[0]); ++n)
	{
		if (CUDA_EXTENSIONS[n]==NULL)
			continue;
		GPU_Extension* gpu_ext = CUDA_EXTENSIONS[n](d, eng_ext, eng);
		if (gpu_ext)
			return gpu_ext;
	}
	return NULL;
}

GPU_MultiGridLink* GPU_Backend_CUDA::CreateMultiGridLink(GPU_Backend* sub_grid, const GPU_MultiGridInterpolation& interpol)
{
	UNUSED(sub_grid);
	UNUSED(interpol);
	return NULL;
}
