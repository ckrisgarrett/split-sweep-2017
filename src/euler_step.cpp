#include "global.h"
#include <math.h>


static double *s_E, *s_J, *s_divJ;


/*
    Extra term for convergence on dt(E)
    This is necessary since there is a source
*/
static
void addConvTerm(double t, double dt, double *E)
{
    for(int i = 0; i < g_nx; i++) {
        double term = sqrt(M_PI) / 4.0 + sqrt(M_PI) / 8.0 * (4.0 * M_PI - 1.0) 
                                       * cos(2.0 * g_x[i] - 2.0 * M_PI * t);
        E[i] += term * dt;
    }
}


/*
    Calculate J
*/
static
void calcJ(double *f, double *J)
{
    // 1D
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx; i++) {
            J[i] = 0.0;
            for(int j = 0; j < g_nv; j++) {
                J[i] += g_dv * g_v[j] * f[IJK(i,j,0)];
            }
        }
    }
    // 2D
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            J[NII(0,i1,i2)] = J[NII(1,i1,i2)] = 0.0;
            for(int j1 = 0; j1 < g_nv; j1++) {
            for(int j2 = 0; j2 < g_nv; j2++) {
                J[NII(0,i1,i2)] += g_dv * g_dv * g_v[j1] * f[IIJJK(i1,i2,j1,j2,0)];
                J[NII(1,i1,i2)] += g_dv * g_dv * g_v[j2] * f[IIJJK(i1,i2,j1,j2,0)];
            }}
        }}
    }
}


/*
    Calculate divergence of J
*/
static
void calcDivJ(double *J, double *divJ)
{
    // 1D
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx; i++) {
            int ia = (i == 0) ? g_nx-1 : i-1;
            int ib = (i == g_nx-1) ? 0 : i+1;
            divJ[IK(i,0)] = (J[ib] - J[ia]) / (2 * g_dx);
            divJ[IK(i,1)] = 0.0;
        }
    }
    // 2D
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            int i1a = (i1 == 0) ? g_nx-1 : i1-1;
            int i1b = (i1 == g_nx-1) ? 0 : i1+1;
            int i2a = (i2 == 0) ? g_nx-1 : i2-1;
            int i2b = (i2 == g_nx-1) ? 0 : i2+1;
            divJ[IIK(i1,i2,0)] = (J[NII(0,i1b,i2)] - J[NII(0,i1a,i2)]) / (2 * g_dx)
                               + (J[NII(1,i1,i2b)] - J[NII(1,i1,i2a)]) / (2 * g_dx);
            divJ[IIK(i1,i2,1)] = 0.0;
            divJ[IIK(i1,i2,2)] = 0.0;
        }}
    }
}


/*
    Allocate memory
*/
void eulerStep_init()
{
    s_J = new double[g_ESize];
    s_E = new double[g_ESize];
    s_divJ = new double[g_rhoSize];
}


/*
    Free memory
*/
void eulerStep_end()
{
    delete[] s_J;
    delete[] s_E;
    delete[] s_divJ;
}


/*
    Takes an euler step.
    Linearizes the problem.  i.e. takes E^{n+1} = E^n + E_t dt
*/
void eulerStep(double t, double dt, double sigma, double *E0, double *q, double *f)
{
    // Set E
    if(g_calcEfield) {
        calcJ(f, s_J);
        
//        for(size_t i = 0; i < g_ESize; i++) {
//            s_E[i] = E0[i] - s_J[i] * dt;
//        }
        
        calcDivJ(s_J, s_divJ);
        efield(s_divJ, s_E);
        for(size_t i = 0; i < g_ESize; i++) {
            s_E[i] = -dt * s_E[i] + E0[i];
        }
        if(g_runType == RUN_CONVERGENCE) {
            addConvTerm(t, dt, s_E);
        }
    }
    else  {
        for(size_t i = 0; i < g_ESize; i++) {
            s_E[i] = E0[i];
        }
    }
    
    
    // Sweep
    if(g_dimension == 1) {
        if(g_useNewSweep)
            sweep1dFast(sigma, s_E, q, f);
        else
            sweep1d(sigma, s_E, q, f);
    }
    else {
        if(g_useNewSweep)
            sweep2dFast(sigma, s_E, q, f);
        else
            sweep2d(sigma, s_E, q, f);
    }
}






