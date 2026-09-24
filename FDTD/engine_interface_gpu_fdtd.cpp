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

#include "engine_interface_gpu_fdtd.h"
#include "engine_gpu.h"

Engine_Interface_GPU_FDTD::Engine_Interface_GPU_FDTD(Operator_GPU* op) : Engine_Interface_FDTD(op)
{
	m_Eng_GPU = dynamic_cast<Engine_GPU*>(op->GetEngine());
}

Engine_Interface_GPU_FDTD::~Engine_Interface_GPU_FDTD()
{
	m_Eng_GPU = NULL;
}

double Engine_Interface_GPU_FDTD::CalcFastEnergy() const
{
	// the end criteria asks for this every Nyquist period, so it is worth the
	// device implementation; without one the host mirror has to be brought up to
	// date, it is not written at the end of a batch
	if (m_Eng_GPU)
	{
		unsigned int numNodes[3];
		for (int n=0; n<3; ++n)
			numNodes[n] = m_Op->GetNumberOfLines(n)-1;
		double E_energy=0.0, H_energy=0.0;
		if (m_Eng_GPU->CalcFastEnergy(numNodes, E_energy, H_energy))
			return EPS0*E_energy + MUE0*H_energy;
		m_Eng_GPU->UpdateHostMirror();
	}
	return Engine_Interface_FDTD::CalcFastEnergy();
}

void Engine_Interface_GPU_FDTD::PrepareFieldAccess()
{
	// several threads read the fields next; reading them from the device one by one is not thread-safe
	if (m_Eng_GPU)
		m_Eng_GPU->UpdateHostMirror();
}

namespace
{
//! Precomputed GetRawInterpolatedField()/GetRawInterpolatedDualField() (type 0) of the
//! nodes of a dump, for fields in the basic engine layout (e.g. the GPU host mirror).
//! Every node component is one of a few forms, evaluated with the same operations as there.
class Field_Gather_FDTD : public Engine_Field_Gather
{
public:
	enum Form { ZERO, RAW, LERP, AVG4 };
	//! one component of a node: raw(k) = value(idx[k])/delta[k] (0 if delta is 0), see GetRawField()
	struct Entry
	{
		Form form;
		unsigned int idx[4];
		double delta[4];
		double rel;      //!< LERP: raw(0)*(1-rel) + raw(1)*rel
	};

	Field_Gather_FDTD(const ArrayLib::ArrayNIJK<FDTD_FLOAT>* volt, const ArrayLib::ArrayNIJK<FDTD_FLOAT>* curr,
	                  bool h_field, unsigned int nj, unsigned int nk)
		: m_Volt(volt), m_Curr(curr), m_H(h_field), m_nj(nj), m_nk(nk) {}

	std::vector<Entry> entries;   //!< [line][k][n]

	virtual void Evaluate(size_t line_start, size_t line_stop, ArrayLib::ArrayNIJK<float> &field, const float* src=NULL) const
	{
		const float* f = src ? src : (m_H ? m_Curr->data() : m_Volt->data());
		for (size_t l=line_start; l<line_stop; ++l)
		{
			const unsigned int i = l/m_nj;
			const unsigned int j = l%m_nj;
			for (unsigned int k=0; k<m_nk; ++k)
				for (int n=0; n<3; ++n)
				{
					const Entry& e = entries[(l*m_nk + k)*3 + n];
					double out = 0;
					switch (e.form)
					{
					case ZERO:
						break;
					case RAW:
						out = Raw(f, e, 0);
						break;
					case LERP:
						out = Raw(f, e, 0)*(1.0-e.rel) + Raw(f, e, 1)*e.rel;
						break;
					case AVG4:
						out = Raw(f, e, 0);
						out+= Raw(f, e, 1);
						out+= Raw(f, e, 2);
						out+= Raw(f, e, 3);
						out/=4;
						break;
					}
					field(n, i, j, k) = out;
				}
		}
	}

protected:
	static inline double Raw(const float* f, const Entry& e, int k)
	{
		double value = f[e.idx[k]];
		if (e.delta[k])
			return value/e.delta[k];
		return 0.0;
	}

	const ArrayLib::ArrayNIJK<FDTD_FLOAT>* m_Volt;
	const ArrayLib::ArrayNIJK<FDTD_FLOAT>* m_Curr;
	bool m_H;
	unsigned int m_nj, m_nk;
};
}
// The same values as GetEField()/GetHField() of Engine_Interface_FDTD: a subclass that
// changes the field evaluation must not use this (see Engine_Interface_Cylindrical_FDTD).
Engine_Field_Gather* Engine_Interface_GPU_FDTD::CreateFieldGather(bool h_field, const unsigned int numLines[3], unsigned int* const posLines[3]) const
{
	if (!m_Eng_GPU)   // another engine on a GPU operator: no gather, the caller reads the fields node by node
		return NULL;

	unsigned int N[3];
	for (int n=0; n<3; ++n)
		N[n] = m_Op->GetNumberOfLines(n, true);
	auto index = [&](int n, const unsigned int* p) -> unsigned int {return ((n*N[0] + p[0])*N[1] + p[1])*N[2] + p[2];};
	auto delta = [&](int n, const unsigned int* p) -> double {return m_Op->GetEdgeLength(n, p, h_field);};

	typedef Field_Gather_FDTD G;
	G* gather = new G(&m_Eng_GPU->HostVoltages(), &m_Eng_GPU->HostCurrents(), h_field, numLines[1], numLines[2]);
	gather->entries.resize((size_t)numLines[0]*numLines[1]*numLines[2]*3);
	size_t e_idx = 0;
	for (unsigned int i=0; i<numLines[0]; ++i)
		for (unsigned int j=0; j<numLines[1]; ++j)
			for (unsigned int k=0; k<numLines[2]; ++k)
			{
				const unsigned int pos[3] = {posLines[0][i], posLines[1][j], posLines[2][k]};
				for (int n=0; n<3; ++n)
				{
					G::Entry& e = gather->entries[e_idx++];
					e.form = G::ZERO;
					e.rel = 0;
					for (int m=0; m<4; ++m) {e.idx[m] = 0; e.delta[m] = 0;}
					unsigned int p[3] = {pos[0], pos[1], pos[2]};
					const int nP = (n+1)%3;
					const int nPP = (n+2)%3;
					auto set = [&](int m) {e.idx[m] = index(n, p); e.delta[m] = delta(n, p);};
					const bool upper = (pos[0]==N[0]-1) || (pos[1]==N[1]-1) || (pos[2]==N[2]-1);
					switch (m_InterpolType)
					{
					default:
					case NO_INTERPOLATION:
						e.form = G::RAW;
						set(0);
						break;
					case NODE_INTERPOLATE:
						if (!h_field)
						{
							// see GetRawInterpolatedField()
							if (pos[n]==N[n]-1)
							{
								--p[n];
								e.form = G::RAW;
								set(0);
								break;
							}
							const double d = delta(n, p);
							if (d==0)
								break;   // ZERO
							e.form = G::RAW;
							set(0);
							if (pos[n]==0)
								break;
							--p[n];
							const double d_down = delta(n, p);
							e.form = G::LERP;
							e.rel = d / (d+d_down);
							set(1);
						}
						else
						{
							// see GetRawInterpolatedDualField()
							if (upper || (pos[nP]==0) || (pos[nPP]==0))
								break;   // ZERO
							e.form = G::AVG4;
							set(0);
							--p[nP];
							set(1);
							--p[nPP];
							set(2);
							++p[nP];
							set(3);
						}
						break;
					case CELL_INTERPOLATE:
						if (!h_field)
						{
							// see GetRawInterpolatedField()
							if (upper)
								break;   // ZERO
							e.form = G::AVG4;
							set(0);
							++p[nP];
							set(1);
							++p[nPP];
							set(2);
							--p[nP];
							set(3);
						}
						else
						{
							// see GetRawInterpolatedDualField()
							if (pos[n]>=N[n]-1)
								break;   // ZERO
							const double d = delta(n, p);
							e.form = G::LERP;
							set(0);
							++p[n];
							const double d_up = delta(n, p);
							e.rel = d / (d+d_up);
							set(1);
						}
						break;
					}
				}
			}
	return gather;
}
