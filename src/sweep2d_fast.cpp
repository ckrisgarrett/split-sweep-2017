/*
    Solves:
    v dx f + a dv f + sigma f = q
*/


#include "global.h"
#include <petscksp.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif


// Variables needed for this file.
static Vec s_bVec, s_fBdryVec;
static KSP s_ksp;
static Mat s_A;
static double *s_fBdry, *s_f;
static double s_sigma, *s_E;
static int s_bdrySize;

static
void mult(Mat A, Vec x, Vec y);


/*
    Initialize solver.
*/
void sweep2dFast_init()
{
    PetscErrorCode err;
    int n;
    
    
    // Size of problem
    n = s_bdrySize = 2 * g_nx * g_nx * g_nv * g_nBasisBdry 
                   + 2 * g_nx * g_nv * g_nv * g_nBasisBdry;
    
    
    // Petsc init
    err = MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, NULL, &s_A);CHKERRV(err);
    err = MatShellSetOperation(s_A, MATOP_MULT, (void (*)(void))mult);CHKERRV(err);
    
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_bVec);CHKERRV(err);
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_fBdryVec);CHKERRV(err);
    
    err = KSPCreate(PETSC_COMM_WORLD, &s_ksp);CHKERRV(err);
    err = KSPSetOperators(s_ksp,s_A,s_A);CHKERRV(err);
    
    err = KSPSetTolerances(s_ksp, g_sweep1Tol, PETSC_DEFAULT, PETSC_DEFAULT, 
                           g_sweep1Maxiter);CHKERRV(err);
    err = KSPSetInitialGuessNonzero(s_ksp, PETSC_TRUE);CHKERRV(err);
    err = KSPSetType(s_ksp, KSPGMRES);CHKERRV(err);
    
    
    // Allocate other data
    s_fBdry = new double[s_bdrySize];
}


/*
    Free solver variables.
*/
void sweep2dFast_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_bVec);CHKERRV(err);
    err = VecDestroy(&s_fBdryVec);CHKERRV(err);
    err = MatDestroy(&s_A);CHKERRV(err);
    err = KSPDestroy(&s_ksp);CHKERRV(err);
    
    delete[] s_fBdry;
}


/*
    Splits a boundary vector used in the GMRES solve into its four components:
    [ax,bx] x [ax,bx] x {0} x [av,bv]
    [ax,bx] x [ax,bx] x [av,bv] x {0}
    {ax} x [ax,bx] x [av,bv] x [av,bv]
    [ax,bx] x {ax} x [av,bv] x [av,bv]
*/
static 
void splitBdry(double *bdry, double **bdryX1, double **bdryX2, 
               double **bdryV1, double **bdryV2)
{
    *bdryX1 = bdry;
    *bdryX2 = *bdryX1 + g_nx * g_nx * g_nv * g_nBasisBdry;
    *bdryV1 = *bdryX2 + g_nx * g_nx * g_nv * g_nBasisBdry;
    *bdryV2 = *bdryV1 + g_nx * g_nv * g_nv * g_nBasisBdry;
}
static 
void splitBdry(const double *bdry, const double **bdryX1, const double **bdryX2, 
               const double **bdryV1, const double **bdryV2)
{
    *bdryX1 = bdry;
    *bdryX2 = *bdryX1 + g_nx * g_nx * g_nv * g_nBasisBdry;
    *bdryV1 = *bdryX2 + g_nx * g_nx * g_nv * g_nBasisBdry;
    *bdryV2 = *bdryV1 + g_nx * g_nv * g_nv * g_nBasisBdry;
}


/*
    Sweep on one cell.
    Note: q and fBdry can be NULL
*/
static
void S1(int i1, int i2, int j1, int j2, double *q, const double *fBdry, double *f)
{
    double rhs[5], lhs[25];
    int nv_m = g_nv / 2 - 1;
    int nv_p = g_nv / 2;
    int diri1, diri2, dirj1, dirj2;
    const double *fBdryX1, *fBdryX2, *fBdryV1, *fBdryV2;
    
    
    #define A(i,j) lhs[((i)*5+(j))]
    
    
    // Determine upwinding direction
    diri1 = (g_v[j1] > 0) ? +1 : -1;
    diri2 = (g_v[j2] > 0) ? +1 : -1;
    
    dirj1 = (s_E[NII(0,i1,i2)] > 0) ? +1 : -1;
    dirj2 = (s_E[NII(1,i1,i2)] > 0) ? +1 : -1;
    
    
    // Split boundary
    if(fBdry == NULL) {
        fBdryX1 = fBdryX2 = fBdryV1 = fBdryV2 = NULL;
    }
    else {
        splitBdry(fBdry, &fBdryX1, &fBdryX2, &fBdryV1, &fBdryV2);
    }
    
    
    // Direction agnostic calculations
    A(0,0) = s_sigma;
    A(1,1) = s_sigma / 3.0;
    A(2,2) = s_sigma / 3.0;
    A(3,3) = s_sigma / 3.0;
    A(4,4) = s_sigma / 3.0;
    A(0,1) = A(0,2) = A(0,3) = A(0,4) = 0.0;
    A(1,0) = A(1,2) = A(1,3) = A(1,4) = 0.0;
    A(2,0) = A(2,1) = A(2,3) = A(2,4) = 0.0;
    A(3,0) = A(3,1) = A(3,2) = A(3,4) = 0.0;
    A(4,0) = A(4,1) = A(4,2) = A(4,3) = 0.0;
    A(1,0) = -2.0 * g_v[j1] / g_dx;
    A(2,0) = -2.0 * g_v[j2] / g_dx;
    A(3,0) = -2.0 * s_E[NII(0,i1,i2)] / g_dv;
    A(4,0) = -2.0 * s_E[NII(1,i1,i2)] / g_dv;
    
    if(q != NULL) {
        rhs[0] = q[IIJJK(i1,i2,j1,j2,0)];
        rhs[1] = q[IIJJK(i1,i2,j1,j2,1)] / 3.0;
        rhs[2] = q[IIJJK(i1,i2,j1,j2,2)] / 3.0;
        rhs[3] = q[IIJJK(i1,i2,j1,j2,3)] / 3.0;
        rhs[4] = q[IIJJK(i1,i2,j1,j2,4)] / 3.0;
    }
    else {
        rhs[0] = 0.0;
        rhs[1] = 0.0;
        rhs[2] = 0.0;
        rhs[3] = 0.0;
        rhs[4] = 0.0;
    }
    
    
    // diri1
    if(diri1 > 0) {
        A(0,0) += g_v[j1] / g_dx;
        A(0,1) += g_v[j1] / g_dx;
        A(1,0) += g_v[j1] / g_dx;
        A(1,1) += g_v[j1] / g_dx;
        A(2,2) += g_v[j1] / g_dx / 3.0;
        A(3,3) += g_v[j1] / g_dx / 3.0;
        A(4,4) += g_v[j1] / g_dx / 3.0;
        
        if(i1 > 0) {
            rhs[0] += g_v[j1] / g_dx 
                * (f[IIJJK(i1-1,i2,j1,j2,0)] + f[IIJJK(i1-1,i2,j1,j2,1)]);
            rhs[1] -= g_v[j1] / g_dx 
                * (f[IIJJK(i1-1,i2,j1,j2,0)] + f[IIJJK(i1-1,i2,j1,j2,1)]);
            rhs[2] += g_v[j1] / g_dx 
                * f[IIJJK(i1-1,i2,j1,j2,2)] / 3.0;
            rhs[3] += g_v[j1] / g_dx 
                * f[IIJJK(i1-1,i2,j1,j2,3)] / 3.0;
            rhs[4] += g_v[j1] / g_dx 
                * f[IIJJK(i1-1,i2,j1,j2,4)] / 3.0;
        }
        
        else if(i1 == 0 && fBdryV1 != NULL) {
            rhs[0] += g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,0)];
            rhs[1] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,0)];
            rhs[2] += g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,1)] / 3.0;
            rhs[3] += g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,2)] / 3.0;
            rhs[4] += g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,3)] / 3.0;
        }
    }
    else if(diri1 < 0) {
        A(0,0) += -g_v[j1] / g_dx;
        A(0,1) +=  g_v[j1] / g_dx;
        A(1,0) +=  g_v[j1] / g_dx;
        A(1,1) += -g_v[j1] / g_dx;
        A(2,2) += -g_v[j1] / g_dx / 3.0;
        A(3,3) += -g_v[j1] / g_dx / 3.0;
        A(4,4) += -g_v[j1] / g_dx / 3.0;
        
        if(i1 < g_nx-1) {
            rhs[0] -= g_v[j1] / g_dx 
                * (f[IIJJK(i1+1,i2,j1,j2,0)] - f[IIJJK(i1+1,i2,j1,j2,1)]);
            rhs[1] -= g_v[j1] / g_dx 
                * (f[IIJJK(i1+1,i2,j1,j2,0)] - f[IIJJK(i1+1,i2,j1,j2,1)]);
            rhs[2] -= g_v[j1] / g_dx 
                * f[IIJJK(i1+1,i2,j1,j2,2)] / 3.0;
            rhs[3] -= g_v[j1] / g_dx 
                * f[IIJJK(i1+1,i2,j1,j2,3)] / 3.0;
            rhs[4] -= g_v[j1] / g_dx 
                * f[IIJJK(i1+1,i2,j1,j2,4)] / 3.0;
        }
        
        else if(i1 == g_nx-1 && fBdryV1 != NULL) {
            rhs[0] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,0)];
            rhs[1] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,0)];
            rhs[2] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,1)] / 3.0;
            rhs[3] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,2)] / 3.0;
            rhs[4] -= g_v[j1] / g_dx * fBdryV1[IJJK(i2,j1,j2,3)] / 3.0;
        }
    }
    
    // diri2
    if(diri2 > 0) {
        A(0,0) += g_v[j2] / g_dx;
        A(0,2) += g_v[j2] / g_dx;
        A(2,0) += g_v[j2] / g_dx;
        A(2,2) += g_v[j2] / g_dx;
        A(1,1) += g_v[j2] / g_dx / 3.0;
        A(3,3) += g_v[j2] / g_dx / 3.0;
        A(4,4) += g_v[j2] / g_dx / 3.0;
        
        if(i2 > 0) {
            rhs[0] += g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2-1,j1,j2,0)] + f[IIJJK(i1,i2-1,j1,j2,2)]);
            rhs[2] -= g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2-1,j1,j2,0)] + f[IIJJK(i1,i2-1,j1,j2,2)]);
            rhs[1] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2-1,j1,j2,1)] / 3.0;
            rhs[3] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2-1,j1,j2,3)] / 3.0;
            rhs[4] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2-1,j1,j2,4)] / 3.0;
        }
        
        else if(i2 == 0 && fBdryV2 != NULL) {
            rhs[0] += g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,0)];
            rhs[2] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,0)];
            rhs[1] += g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,1)] / 3.0;
            rhs[3] += g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,2)] / 3.0;
            rhs[4] += g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,3)] / 3.0;
        }
    }
    else if(diri2 < 0) {
        A(0,0) += -g_v[j2] / g_dx;
        A(0,2) +=  g_v[j2] / g_dx;
        A(2,0) +=  g_v[j2] / g_dx;
        A(2,2) += -g_v[j2] / g_dx;
        A(1,1) += -g_v[j2] / g_dx / 3.0;
        A(3,3) += -g_v[j2] / g_dx / 3.0;
        A(4,4) += -g_v[j2] / g_dx / 3.0;
        
        if(i2 < g_nx-1) {
            rhs[0] -= g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2+1,j1,j2,0)] - f[IIJJK(i1,i2+1,j1,j2,2)]);
            rhs[2] -= g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2+1,j1,j2,0)] - f[IIJJK(i1,i2+1,j1,j2,2)]);
            rhs[1] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2+1,j1,j2,1)] / 3.0;
            rhs[3] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2+1,j1,j2,3)] / 3.0;
            rhs[4] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2+1,j1,j2,4)] / 3.0;
        }
        
        else if(i2 == g_nx-1 && fBdryV2 != NULL) {
            rhs[0] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,0)];
            rhs[2] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,0)];
            rhs[1] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,1)] / 3.0;
            rhs[3] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,2)] / 3.0;
            rhs[4] -= g_v[j2] / g_dx * fBdryV2[IJJK(i1,j1,j2,3)] / 3.0;
        }
    }
    
    // dirj1
    if(dirj1 > 0) {
        A(0,0) += s_E[NII(0,i1,i2)] / g_dv;
        A(0,3) += s_E[NII(0,i1,i2)] / g_dv;
        A(3,0) += s_E[NII(0,i1,i2)] / g_dv;
        A(3,3) += s_E[NII(0,i1,i2)] / g_dv;
        A(1,1) += s_E[NII(0,i1,i2)] / g_dv / 3.0;
        A(2,2) += s_E[NII(0,i1,i2)] / g_dv / 3.0;
        A(4,4) += s_E[NII(0,i1,i2)] / g_dv / 3.0;
        
        if(j1 > 0 && j1 != nv_p) {
            rhs[0] += s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1-1,j2,0)] + f[IIJJK(i1,i2,j1-1,j2,3)]);
            rhs[3] -= s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1-1,j2,0)] + f[IIJJK(i1,i2,j1-1,j2,3)]);
            rhs[1] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1-1,j2,1)] / 3.0;
            rhs[2] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1-1,j2,2)] / 3.0;
            rhs[4] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1-1,j2,4)] / 3.0;
        }
        
        else if(j1 == nv_p && fBdryX1 != NULL) {
            rhs[0] += s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,0)];
            rhs[3] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,0)];
            rhs[1] += s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,1)] / 3.0;
            rhs[2] += s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,2)] / 3.0;
            rhs[4] += s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,3)] / 3.0;
        }
    }
    else if(dirj1 < 0) {
        A(0,0) += -s_E[NII(0,i1,i2)] / g_dv;
        A(0,3) +=  s_E[NII(0,i1,i2)] / g_dv;
        A(3,0) +=  s_E[NII(0,i1,i2)] / g_dv;
        A(3,3) += -s_E[NII(0,i1,i2)] / g_dv;
        A(1,1) += -s_E[NII(0,i1,i2)] / g_dv / 3.0;
        A(2,2) += -s_E[NII(0,i1,i2)] / g_dv / 3.0;
        A(4,4) += -s_E[NII(0,i1,i2)] / g_dv / 3.0;
        
        if(j1 < g_nv-1 && j1 != nv_m) {
            rhs[0] -= s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1+1,j2,0)] - f[IIJJK(i1,i2,j1+1,j2,3)]);
            rhs[3] -= s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1+1,j2,0)] - f[IIJJK(i1,i2,j1+1,j2,3)]);
            rhs[1] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1+1,j2,1)] / 3.0;
            rhs[2] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1+1,j2,2)] / 3.0;
            rhs[4] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1+1,j2,4)] / 3.0;
        }
        
        else if(j1 == nv_m && fBdryX1 != NULL) {
            rhs[0] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,0)];
            rhs[3] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,0)];
            rhs[1] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,1)] / 3.0;
            rhs[2] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,2)] / 3.0;
            rhs[4] -= s_E[NII(0,i1,i2)] / g_dv * fBdryX1[IIJK(i1,i2,j2,3)] / 3.0;
        }
    }
    
    // dirj2
    if(dirj2 > 0) {
        A(0,0) += s_E[NII(1,i1,i2)] / g_dv;
        A(0,4) += s_E[NII(1,i1,i2)] / g_dv;
        A(4,0) += s_E[NII(1,i1,i2)] / g_dv;
        A(4,4) += s_E[NII(1,i1,i2)] / g_dv;
        A(1,1) += s_E[NII(1,i1,i2)] / g_dv / 3.0;
        A(2,2) += s_E[NII(1,i1,i2)] / g_dv / 3.0;
        A(3,3) += s_E[NII(1,i1,i2)] / g_dv / 3.0;
        
        if(j2 > 0 && j2 != nv_p) {
            rhs[0] += s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2-1,0)] + f[IIJJK(i1,i2,j1,j2-1,4)]);
            rhs[4] -= s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2-1,0)] + f[IIJJK(i1,i2,j1,j2-1,4)]);
            rhs[1] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2-1,1)] / 3.0;
            rhs[2] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2-1,2)] / 3.0;
            rhs[3] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2-1,3)] / 3.0;
        }
        
        else if(j2 == nv_p && fBdryX2 != NULL) {
            rhs[0] += s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,0)];
            rhs[4] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,0)];
            rhs[1] += s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,1)] / 3.0;
            rhs[2] += s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,2)] / 3.0;
            rhs[3] += s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,3)] / 3.0;
        }
    }
    else if(dirj2 < 0) {
        A(0,0) += -s_E[NII(1,i1,i2)] / g_dv;
        A(0,4) +=  s_E[NII(1,i1,i2)] / g_dv;
        A(4,0) +=  s_E[NII(1,i1,i2)] / g_dv;
        A(4,4) += -s_E[NII(1,i1,i2)] / g_dv;
        A(1,1) += -s_E[NII(1,i1,i2)] / g_dv / 3.0;
        A(2,2) += -s_E[NII(1,i1,i2)] / g_dv / 3.0;
        A(3,3) += -s_E[NII(1,i1,i2)] / g_dv / 3.0;
        
        if(j2 < g_nv-1 && j2 != nv_m) {
            rhs[0] -= s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2+1,0)] - f[IIJJK(i1,i2,j1,j2+1,4)]);
            rhs[4] -= s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2+1,0)] - f[IIJJK(i1,i2,j1,j2+1,4)]);
            rhs[1] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2+1,1)] / 3.0;
            rhs[2] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2+1,2)] / 3.0;
            rhs[3] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2+1,3)] / 3.0;
        }
        
        else if(j2 == nv_m && fBdryX2 != NULL) {
            rhs[0] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,0)];
            rhs[4] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,0)];
            rhs[1] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,1)] / 3.0;
            rhs[2] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,2)] / 3.0;
            rhs[3] -= s_E[NII(1,i1,i2)] / g_dv * fBdryX2[IIJK(i1,i2,j1,3)] / 3.0;
        }
    }
    
    
    // Solve the local system
    solveNxN(5, lhs, rhs);
    
    f[IIJJK(i1,i2,j1,j2,0)] = rhs[0];
    f[IIJJK(i1,i2,j1,j2,1)] = rhs[1];
    f[IIJJK(i1,i2,j1,j2,2)] = rhs[2];
    f[IIJJK(i1,i2,j1,j2,3)] = rhs[3];
    f[IIJJK(i1,i2,j1,j2,4)] = rhs[4];
    
    #undef A
}


/*
    Perform the sweep
    Note: q or fBdry may be NULL
*/
static
void S(double *q, const double *fBdry, double *f)
{
    int nv_m = g_nv / 2 - 1;
    int nv_p = g_nv / 2;
    
    // Indices for subdomains: v1 > 0, v2 > 0
    //                         v1 > 0, v2 < 0
    //                         v1 < 0, v2 > 0
    //                         v1 < 0, v2 < 0
    int begin_i1[4] = {0, 0, g_nx-1, g_nx-1};
    int end_i1[4] = {g_nx, g_nx, -1, -1};
    int inc_i1[4] = {1, 1, -1, -1};
    int begin_i2[4] = {0, g_nx-1, 0, g_nx-1};
    int end_i2[4] = {g_nx, -1, g_nx, -1};
    int inc_i2[4] = {1, -1, 1, -1};
    
    
    #pragma omp parallel for schedule(static,1)
    for(int iIndex = 0; iIndex < 4; iIndex++) {
        for(int i1 = begin_i1[iIndex]; i1 != end_i1[iIndex]; i1 = i1 + inc_i1[iIndex]) {
        for(int i2 = begin_i2[iIndex]; i2 != end_i2[iIndex]; i2 = i2 + inc_i2[iIndex]) {
            
            // Bounds of subdomain
            int low1  = (inc_i1[iIndex] == 1) ? nv_p : 0;
            int high1 = (inc_i1[iIndex] == 1) ? g_nx-1 : nv_m;
            int low2  = (inc_i2[iIndex] == 1) ? nv_p : 0;
            int high2 = (inc_i2[iIndex] == 1) ? g_nx-1 : nv_m;
            
            // Direction of sweep
            int begin_j1 = (s_E[NII(0,i1,i2)] > 0) ?    low1 :   high1;
            int end_j1   = (s_E[NII(0,i1,i2)] > 0) ? high1+1 :  low1-1;
            int inc_j1   = (s_E[NII(0,i1,i2)] > 0) ?       1 :      -1;
            int begin_j2 = (s_E[NII(1,i1,i2)] > 0) ?    low2 :   high2;
            int end_j2   = (s_E[NII(1,i1,i2)] > 0) ? high2+1 :  low2-1;
            int inc_j2   = (s_E[NII(1,i1,i2)] > 0) ?       1 :      -1;
            
            // Sweep
            for(int j1 = begin_j1; j1 != end_j1; j1 = j1 + inc_j1) {
            for(int j2 = begin_j2; j2 != end_j2; j2 = j2 + inc_j2) {
                S1(i1, i2, j1, j2, q, fBdry, f);
            }}
        }}
    }
}


/*
    Project to boundary
*/
static
void P(double *f, double *fBdry)
{
    int nv_m = g_nv / 2 - 1;
    int nv_p = g_nv / 2;
    double *fBdryX1, *fBdryX2, *fBdryV1, *fBdryV2;
    
    
    // Split boundary vector
    splitBdry(fBdry, &fBdryX1, &fBdryX2, &fBdryV1, &fBdryV2);
    
    
    // Boundary X1, i.e. [ax,bx] x [ax,bx] x {0} x [av,bv]
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j = 0; j < g_nv; j++) {
        if(s_E[NII(0,i1,i2)] > 0) {
            fBdryX1[IIJK(i1,i2,j,0)] = f[IIJJK(i1,i2,nv_m,j,0)] 
                                     + f[IIJJK(i1,i2,nv_m,j,3)];
            fBdryX1[IIJK(i1,i2,j,1)] = f[IIJJK(i1,i2,nv_m,j,1)];
            fBdryX1[IIJK(i1,i2,j,2)] = f[IIJJK(i1,i2,nv_m,j,2)];
            fBdryX1[IIJK(i1,i2,j,3)] = f[IIJJK(i1,i2,nv_m,j,4)];
        }
        else {
            fBdryX1[IIJK(i1,i2,j,0)] = f[IIJJK(i1,i2,nv_p,j,0)] 
                                     - f[IIJJK(i1,i2,nv_p,j,3)];
            fBdryX1[IIJK(i1,i2,j,1)] = f[IIJJK(i1,i2,nv_p,j,1)];
            fBdryX1[IIJK(i1,i2,j,2)] = f[IIJJK(i1,i2,nv_p,j,2)];
            fBdryX1[IIJK(i1,i2,j,3)] = f[IIJJK(i1,i2,nv_p,j,4)];
        }
    }}}
    
    
    // Boundary X2, i.e. [ax,bx] x [ax,bx] x [av,bv] x {0}
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j = 0; j < g_nv; j++) {
        if(s_E[NII(1,i1,i2)] > 0) {
            fBdryX2[IIJK(i1,i2,j,0)] = f[IIJJK(i1,i2,j,nv_m,0)] 
                                     + f[IIJJK(i1,i2,j,nv_m,4)];
            fBdryX2[IIJK(i1,i2,j,1)] = f[IIJJK(i1,i2,j,nv_m,1)];
            fBdryX2[IIJK(i1,i2,j,2)] = f[IIJJK(i1,i2,j,nv_m,2)];
            fBdryX2[IIJK(i1,i2,j,3)] = f[IIJJK(i1,i2,j,nv_m,3)];
        }
        else {
            fBdryX2[IIJK(i1,i2,j,0)] = f[IIJJK(i1,i2,j,nv_p,0)] 
                                     - f[IIJJK(i1,i2,j,nv_p,4)];
            fBdryX2[IIJK(i1,i2,j,1)] = f[IIJJK(i1,i2,j,nv_p,1)];
            fBdryX2[IIJK(i1,i2,j,2)] = f[IIJJK(i1,i2,j,nv_p,2)];
            fBdryX2[IIJK(i1,i2,j,3)] = f[IIJJK(i1,i2,j,nv_p,3)];
        }
    }}}
    
    
    // Boundary V1, i.e. {ax} x [ax,bx] x [av,bv] x [av,bv]
    for(int i = 0; i < g_nx; i++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        if(g_isPeriodic) {
            if(g_v[j1] > 0) {
                fBdryV1[IJJK(i,j1,j2,0)] = f[IIJJK(g_nx-1,i,j1,j2,0)] 
                                         + f[IIJJK(g_nx-1,i,j1,j2,1)];
                fBdryV1[IJJK(i,j1,j2,1)] = f[IIJJK(g_nx-1,i,j1,j2,2)];
                fBdryV1[IJJK(i,j1,j2,2)] = f[IIJJK(g_nx-1,i,j1,j2,3)];
                fBdryV1[IJJK(i,j1,j2,3)] = f[IIJJK(g_nx-1,i,j1,j2,4)];
            }
            else {
                fBdryV1[IJJK(i,j1,j2,0)] = f[IIJJK(0,i,j1,j2,0)] 
                                         - f[IIJJK(0,i,j1,j2,1)];
                fBdryV1[IJJK(i,j1,j2,1)] = f[IIJJK(0,i,j1,j2,2)];
                fBdryV1[IJJK(i,j1,j2,2)] = f[IIJJK(0,i,j1,j2,3)];
                fBdryV1[IJJK(i,j1,j2,3)] = f[IIJJK(0,i,j1,j2,4)];
            }
        }
        else {
            fBdryV1[IJJK(i,j1,j2,0)] = 0.0;
            fBdryV1[IJJK(i,j1,j2,1)] = 0.0;
            fBdryV1[IJJK(i,j1,j2,2)] = 0.0;
            fBdryV1[IJJK(i,j1,j2,3)] = 0.0;
        }
    }}}
    
    
    // Boundary V2, i.e. [ax,bx] x {ax} x [av,bv] x [av,bv]
    for(int i = 0; i < g_nx; i++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        if(g_isPeriodic) {
            if(g_v[j2] > 0) {
                fBdryV2[IJJK(i,j1,j2,0)] = f[IIJJK(i,g_nx-1,j1,j2,0)] 
                                         + f[IIJJK(i,g_nx-1,j1,j2,2)];
                fBdryV2[IJJK(i,j1,j2,1)] = f[IIJJK(i,g_nx-1,j1,j2,1)];
                fBdryV2[IJJK(i,j1,j2,2)] = f[IIJJK(i,g_nx-1,j1,j2,3)];
                fBdryV2[IJJK(i,j1,j2,3)] = f[IIJJK(i,g_nx-1,j1,j2,4)];
            }
            else {
                fBdryV2[IJJK(i,j1,j2,0)] = f[IIJJK(i,0,j1,j2,0)] 
                                         - f[IIJJK(i,0,j1,j2,2)];
                fBdryV2[IJJK(i,j1,j2,1)] = f[IIJJK(i,0,j1,j2,1)];
                fBdryV2[IJJK(i,j1,j2,2)] = f[IIJJK(i,0,j1,j2,3)];
                fBdryV2[IJJK(i,j1,j2,3)] = f[IIJJK(i,0,j1,j2,4)];
            }
        }
        else {
            fBdryV2[IJJK(i,j1,j2,0)] = 0.0;
            fBdryV2[IJJK(i,j1,j2,1)] = 0.0;
            fBdryV2[IJJK(i,j1,j2,2)] = 0.0;
            fBdryV2[IJJK(i,j1,j2,3)] = 0.0;
        }
    }}}
}


/*
    Matrix to apply for GMRES
*/
static
void mult(Mat A, Vec x, Vec y)
{
    const double *fBdry;
    double *Af;
    PetscErrorCode err;
    
    
    // Get arrays from PETSC
    err = VecGetArrayRead(x, &fBdry);CHKERRV(err);
    err = VecGetArray(y, &Af);CHKERRV(err);
    
    
    // Save the initial guess for the boundary
    for(int index = 0; index < s_bdrySize; index++) {
        Af[index] = fBdry[index];
    }
    
    
    // Get f from boundary neglecting source
    S(NULL, fBdry, s_f);
    
    
    // Get boundary from f
    P(s_f, s_fBdry);
    
    
    // Perform (I - PA) f_b
    for(int index = 0; index < s_bdrySize; index++) {
        Af[index] = Af[index] - s_fBdry[index];
    }
    
    
    // Give array back to PETSC
    err = VecRestoreArrayRead(x, &fBdry);CHKERRV(err);
    err = VecRestoreArray(y, &Af);CHKERRV(err);
}


/*
    Sweep Function
*/
void sweep2dFast(double sigma, double *E, double *q, double *f)
{
    int iter;
    PetscErrorCode err;
    double *fBdry;
    double *b;
    double *r_q;
    KSPConvergedReason reason;
    
    
    // To be used later.
    // f is set at the very end, so we can abuse its memory here.
    // Allows not wasting memory.
    s_f = f;
    r_q = f;
    
    
    // Set static variables
    s_sigma = sigma;
    s_E = E;
    
    
    // Estimate PS norm is requested
    if(g_estimatePSNorm) {
        estimateNormPS2d();
        exit(0);
    }
    
    
    // Initial guess for boundary (use current boundary)
    err = VecGetArray(s_fBdryVec, &fBdry);CHKERRV(err);
    P(f, fBdry);
    err = VecRestoreArray(s_fBdryVec, &fBdry);CHKERRV(err);
    
    
    // Setup RHS
    err = VecGetArray(s_bVec, &b);CHKERRV(err);
    S(q, NULL, r_q);
    P(r_q, b);
    err = VecRestoreArray(s_bVec, &b);CHKERRV(err);
    
    
    // GMRES
    err = KSPSolve(s_ksp, s_bVec, s_fBdryVec);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp, &iter);CHKERRV(err);
    
    
    // One last sweep to get f
    err = VecGetArray(s_fBdryVec, &fBdry);CHKERRV(err);
    S(q, fBdry, f);
    err = VecRestoreArray(s_fBdryVec, &fBdry);CHKERRV(err);
    
    
    // Print and save sweep iters
    err = KSPGetConvergedReason(s_ksp, &reason);CHKERRV(err);
    if(g_printSweep)
        printf("   sweep2dFast: GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}


extern "C"
double dnrm2_(int *n, double *x, int *incx);

void estimateNormPS2d()
{
    const int maxiter = 50;
    int incx = 1;
    double norm;
    int iter = 0;
    
    
    // Initial condition
    for(int i = 0; i < s_bdrySize; i++) {
        s_fBdry[i] = 1.0;
    }
    
    // Calculate norm to begin algorithm
    norm = dnrm2_(&s_bdrySize, s_fBdry, &incx);
    
    while(iter < maxiter) {
        
        // Normalize vector
        for(int i = 0; i < s_bdrySize; i++) {
            s_fBdry[i] = s_fBdry[i] / norm;
        }
        
        // Perform PS operator
        S(NULL, s_fBdry, s_f);
        P(s_f, s_fBdry);
        
        // Get Norm
        norm = dnrm2_(&s_bdrySize, s_fBdry, &incx);
        printf("Norm: %e\n", norm);
        
        // Next iteration
        iter++;
    }
}




