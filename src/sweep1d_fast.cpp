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
    Allocate static variables.
*/
void sweep1dFast_init()
{
    int n;
    PetscErrorCode err;
    
    
    // Size of problem
    n = s_bdrySize = g_nx * g_nBasisBdry + g_nv * g_nBasisBdry;
    
    
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
    Free static variables.
*/
void sweep1dFast_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_bVec);CHKERRV(err);
    err = VecDestroy(&s_fBdryVec);CHKERRV(err);
    err = MatDestroy(&s_A);CHKERRV(err);
    err = KSPDestroy(&s_ksp);CHKERRV(err);
    
    delete[] s_fBdry;
}


/*
    Splits a boundary vector used in the GMRES solve into its two components:
    [ax,bx] x {0}
    {ax} x [av,bv]
*/
static 
void splitBdry(double *bdry, double **bdryX, double **bdryV)
{
    *bdryX = bdry;
    *bdryV = *bdryX + g_nx * g_nBasisBdry;
}
static 
void splitBdry(const double *bdry, const double **bdryX, const double **bdryV)
{
    *bdryX = bdry;
    *bdryV = *bdryX + g_nx * g_nBasisBdry;
}


/*
    Sweep on one cell.
    Note: q and fBdry can be NULL
*/
static
void S1(int i, int j, double *q, const double *fBdry, double *f)
{
    double rhs[3], lhs[9];
    int nv_m = g_nv / 2 - 1;
    int nv_p = g_nv / 2;
    int diri, dirj;
    const double *fBdryX, *fBdryV;
    
    
    #define A(i,j) lhs[((i)*3+(j))]
    
    
    // Determine upwinding direction
    diri = (g_v[j] > 0) ? +1 : -1;
    dirj = (s_E[i] > 0) ? +1 : -1;
    
    
    // Split boundary
    if(fBdry == NULL) {
        fBdryX = fBdryV = NULL;
    }
    else {
        splitBdry(fBdry, &fBdryX, &fBdryV);
    }
    
    
    // Direction agnostic calculations
    A(0,0) = s_sigma;
    A(1,1) = s_sigma / 3.0;
    A(2,2) = s_sigma / 3.0;
    A(0,1) = A(0,2) = A(1,0) = A(1,2) = A(2,0) = A(2,1) = 0.0;
    
    if(q != NULL) {
        rhs[0] = q[IJK(i,j,0)];
        rhs[1] = q[IJK(i,j,1)] / 3.0;
        rhs[2] = q[IJK(i,j,2)] / 3.0;
    }
    else {
        rhs[0] = rhs[1] = rhs[2] = 0.0;
    }
    
    
    // diri
    if(diri > 0) {
        A(0,0) +=  g_v[j] / g_dx;
        A(0,1) +=  g_v[j] / g_dx;
        A(1,0) += -g_v[j] / g_dx;
        A(1,1) +=  g_v[j] / g_dx;
        A(2,2) +=  g_v[j] / g_dx / 3.0;
        
        if(i > 0) {
            rhs[0] += g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
            rhs[1] -= g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
            rhs[2] += g_v[j] / g_dx * f[IJK(i-1,j,2)] / 3.0;
        }
        
        else if(i == 0 && fBdryV != NULL) {
            rhs[0] += g_v[j] / g_dx * fBdryV[JK(j,0)];
            rhs[1] -= g_v[j] / g_dx * fBdryV[JK(j,0)];
            rhs[2] += g_v[j] / g_dx * fBdryV[JK(j,1)] / 3.0;
        }
    }
    else if(diri < 0) {
        A(0,0) += -g_v[j] / g_dx;
        A(0,1) +=  g_v[j] / g_dx;
        A(1,0) += -g_v[j] / g_dx;
        A(1,1) += -g_v[j] / g_dx;
        A(2,2) += -g_v[j] / g_dx / 3.0;
        
        if(i < g_nx - 1) {
            rhs[0] -= g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
            rhs[1] -= g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
            rhs[2] -= g_v[j] / g_dx * f[IJK(i+1,j,2)] / 3.0;
        }
        
        else if(i == g_nx - 1 && fBdryV != NULL) {
            rhs[0] -= g_v[j] / g_dx * fBdryV[JK(j,0)];
            rhs[1] -= g_v[j] / g_dx * fBdryV[JK(j,0)];
            rhs[2] -= g_v[j] / g_dx * fBdryV[JK(j,1)] / 3.0;
        }
    }
    
    // dirj
    if(dirj > 0) {
        A(0,0) +=  s_E[i] / g_dv;
        A(0,2) +=  s_E[i] / g_dv;
        A(1,1) +=  s_E[i] / g_dv / 3.0;
        A(2,0) += -s_E[i] / g_dv;
        A(2,2) +=  s_E[i] / g_dv;
        
        if(j > 0 && j != nv_p) {
            rhs[0] += s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
            rhs[1] += s_E[i] / g_dv * f[IJK(i,j-1,1)] / 3.0;
            rhs[2] -= s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
        }
        
        else if(j == nv_p && fBdryX != NULL) {
            rhs[0] += s_E[i] / g_dv * fBdryX[IK(i,0)];
            rhs[1] += s_E[i] / g_dv * fBdryX[IK(i,1)] / 3.0;
            rhs[2] -= s_E[i] / g_dv * fBdryX[IK(i,0)];
        }
    }
    else if(dirj < 0) {
        A(0,0) += -s_E[i] / g_dv;
        A(0,2) +=  s_E[i] / g_dv;
        A(1,1) += -s_E[i] / g_dv / 3.0;
        A(2,0) += -s_E[i] / g_dv;
        A(2,2) += -s_E[i] / g_dv;
        
        if(j < g_nv - 1 && j != nv_m) {
            rhs[0] -= s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
            rhs[1] -= s_E[i] / g_dv * f[IJK(i,j+1,1)] / 3.0;
            rhs[2] -= s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
        }
        
        else if(j == nv_m && fBdryX != NULL) {
            rhs[0] -= s_E[i] / g_dv * fBdryX[IK(i,0)];
            rhs[1] -= s_E[i] / g_dv * fBdryX[IK(i,1)] / 3.0;
            rhs[2] -= s_E[i] / g_dv * fBdryX[IK(i,0)];
        }
    }
    
    
    // Solve the local system
    solveNxN(3, lhs, rhs);
    
    f[IJK(i,j,0)] = rhs[0];
    f[IJK(i,j,1)] = rhs[1];
    f[IJK(i,j,2)] = rhs[2];
    
    #undef A
}


/*
    Do the sweep
    Note: q or fBdry may be NULL
*/
static
void S(double *q, const double *fBdry, double *f)
{
    int nv_m = g_nv / 2 - 1;
    int nv_p = g_nv / 2;
    
    // Indices for subdomains: v > 0 and v < 0
    int begin_i[2] = {0, g_nx-1};
    int end_i[2] = {g_nx, -1};
    int inc_i[2] = {1, -1};
    
    #pragma omp parallel for schedule(static,1)
    for(int iIndex = 0; iIndex < 2; iIndex++) {
        for(int i = begin_i[iIndex]; i != end_i[iIndex]; i = i + inc_i[iIndex]) {
            
            // Bounds of subdomain
            int low = (inc_i[iIndex] == 1) ? nv_p : 0;
            int high = (inc_i[iIndex] == 1) ? g_nx-1 : nv_m;
            
            // Direction of sweep
            int begin_j = (s_E[i] > 0) ? low : high;
            int end_j   = (s_E[i] > 0) ? high+1 : low-1;
            int inc_j   = (s_E[i] > 0) ? 1 : -1;
            
            for(int j = begin_j; j != end_j; j = j + inc_j) {
                S1(i, j, q, fBdry, f);
            }
        }
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
    double *fBdryX, *fBdryV;
    
    
    // Split boundary vector
    splitBdry(fBdry, &fBdryX, &fBdryV);
    
    
    // Boundary X, i.e. [ax,bx] x {0}
    for(int i = 0; i < g_nx; i++) {
        if(s_E[i] > 0) {
            fBdryX[IK(i,0)] = f[IJK(i,nv_m,0)] + f[IJK(i,nv_m,2)];
            fBdryX[IK(i,1)] = f[IJK(i,nv_m,1)];
        }
        else {
            fBdryX[IK(i,0)] = f[IJK(i,nv_p,0)] - f[IJK(i,nv_p,2)];
            fBdryX[IK(i,1)] = f[IJK(i,nv_p,1)];
        }
    }
    
    
    // Boundary V, i.e. {ax} x [av,bv]
    for(int j = 0; j < g_nv; j++) {
        if(g_isPeriodic) {
            if(g_v[j] > 0) {
                fBdryV[JK(j,0)] = f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)];
                fBdryV[JK(j,1)] = f[IJK(g_nx-1,j,1)];
            }
            else {
                fBdryV[JK(j,0)] = f[IJK(0,j,0)] - f[IJK(0,j,1)];
                fBdryV[JK(j,1)] = f[IJK(0,j,1)];
            }
        }
        else {
            fBdryV[JK(j,0)] = 0.0;
            fBdryV[JK(j,1)] = 0.0;
        }
    }
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
void sweep1dFast(double sigma, double *E, double *q, double *f)
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
        estimateNormPS1d();
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
        printf("   sweep1dFast: GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}


extern "C"
double dnrm2_(int *n, double *x, int *incx);

void estimateNormPS1d()
{
    const int maxiter = 10;
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





