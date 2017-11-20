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


// Variables needed for this file.
static KSP s_ksp;
static Mat s_A;
static Vec s_b, s_f;
static double s_sigma, *s_E;

static
void mult(Mat A, Vec x, Vec y);


/*
    Preconditioning function
    Implements fast upwind sweep as a preconditioner for the central sweep.
*/
static
PetscErrorCode pcFunction(PC pc, Vec x, Vec y)
{
    PetscErrorCode err = 0;
    double *f;
    double *temp;
    
    static bool firstTime = true;
    static double *q = NULL;
    if(firstTime) {
        int n = g_nx * g_nx * g_nv * g_nv * g_nBasis;
        
        firstTime = false;
        q = new double[n];
    }
    
    VecCopy(x, y);
    err = VecGetArray(x, &temp);CHKERRQ(err);
    err = VecGetArray(y, &f);CHKERRQ(err);
    
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        q[IIJJK(i1,i2,j1,j2,0)] = temp[IIJJK(i1,i2,j1,j2,0)];
        q[IIJJK(i1,i2,j1,j2,1)] = temp[IIJJK(i1,i2,j1,j2,1)] * 3.0;
        q[IIJJK(i1,i2,j1,j2,2)] = temp[IIJJK(i1,i2,j1,j2,2)] * 3.0;
        q[IIJJK(i1,i2,j1,j2,3)] = temp[IIJJK(i1,i2,j1,j2,3)] * 3.0;
        q[IIJJK(i1,i2,j1,j2,4)] = temp[IIJJK(i1,i2,j1,j2,4)] * 3.0;
    }}}}
    sweep2dFast(s_sigma, s_E, q, f);
    
    err = VecRestoreArray(y, &f);CHKERRQ(err);
    err = VecRestoreArray(x, &temp);CHKERRQ(err);
    
    return err;
}


// Allocate static variables.
void sweep2d_init()
{
    int n = g_fSize;
    PetscErrorCode err;
    
    // Petsc init
    err = MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, NULL, &s_A);CHKERRV(err);
    err = MatShellSetOperation(s_A, MATOP_MULT, (void (*)(void))mult);CHKERRV(err);
    
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_b);CHKERRV(err);
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_f);CHKERRV(err);
    
    err = KSPCreate(PETSC_COMM_WORLD, &s_ksp);CHKERRV(err);
    err = KSPSetOperators(s_ksp,s_A,s_A);CHKERRV(err);
    
    err = KSPSetTolerances(s_ksp, g_sweepTol, PETSC_DEFAULT, PETSC_DEFAULT, 
                           g_sweepMaxiter);CHKERRV(err);
    err = KSPSetInitialGuessNonzero(s_ksp, PETSC_TRUE);CHKERRV(err);
    err = KSPSetType(s_ksp, KSPGMRES);CHKERRV(err);
    
    // Preconditioner setup
    if(g_precondition) {
        PC pc; 
        err = KSPSetPCSide(s_ksp, PC_RIGHT);CHKERRV(err);
        err = KSPGetPC(s_ksp, &pc);CHKERRV(err);
        err = PCSetType(pc, PCSHELL);CHKERRV(err);
        err = PCShellSetApply(pc, pcFunction);CHKERRV(err);
    }
}


// Free static variables.
void sweep2d_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_b);CHKERRV(err);
    err = VecDestroy(&s_f);CHKERRV(err);
    err = MatDestroy(&s_A);CHKERRV(err);
    err = KSPDestroy(&s_ksp);CHKERRV(err);
}



// Matrix to apply for GMRES
static
void mult(Mat A, Vec x, Vec y)
{
    const double *f;
    double *Af;
    PetscErrorCode err;
    
    err = VecGetArrayRead(x, &f);CHKERRV(err);
    err = VecGetArray(y, &Af);CHKERRV(err);
    
    // Do an iteration.
    #pragma omp parallel for
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        
        // Diagonal Term
        Af[IIJJK(i1,i2,j1,j2,0)] = s_sigma * f[IIJJK(i1,i2,j1,j2,0)];
        Af[IIJJK(i1,i2,j1,j2,1)] = s_sigma / 3.0 * f[IIJJK(i1,i2,j1,j2,1)];
        Af[IIJJK(i1,i2,j1,j2,2)] = s_sigma / 3.0 * f[IIJJK(i1,i2,j1,j2,2)];
        Af[IIJJK(i1,i2,j1,j2,3)] = s_sigma / 3.0 * f[IIJJK(i1,i2,j1,j2,3)];
        Af[IIJJK(i1,i2,j1,j2,4)] = s_sigma / 3.0 * f[IIJJK(i1,i2,j1,j2,4)];
        
        Af[IIJJK(i1,i2,j1,j2,1)] -= 2.0 * g_v[j1] / g_dx * f[IIJJK(i1,i2,j1,j2,0)];
        Af[IIJJK(i1,i2,j1,j2,2)] -= 2.0 * g_v[j2] / g_dx * f[IIJJK(i1,i2,j1,j2,0)];
        Af[IIJJK(i1,i2,j1,j2,3)] -= 2.0 * s_E[NII(0,i1,i2)] / g_dv * f[IIJJK(i1,i2,j1,j2,0)];
        Af[IIJJK(i1,i2,j1,j2,4)] -= 2.0 * s_E[NII(1,i1,i2)] / g_dv * f[IIJJK(i1,i2,j1,j2,0)];
        
        
        // Off diagonal i1-Term
        if(g_upwindFlux && g_v[j1] > 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j1] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,1)]);
            Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,1)]);
            Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(i1 > 0) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j1] / g_dx 
                    * (f[IIJJK(i1-1,i2,j1,j2,0)] + f[IIJJK(i1-1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                    * (f[IIJJK(i1-1,i2,j1,j2,0)] + f[IIJJK(i1-1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,2)] -= g_v[j1] / g_dx 
                    * f[IIJJK(i1-1,i2,j1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j1] / g_dx 
                    * f[IIJJK(i1-1,i2,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j1] / g_dx 
                    * f[IIJJK(i1-1,i2,j1,j2,4)] / 3.0;
            }
            
            if(i1 == 0 && g_isPeriodic) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j1] / g_dx 
                    * (f[IIJJK(g_nx-1,i2,j1,j2,0)] + f[IIJJK(g_nx-1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                    * (f[IIJJK(g_nx-1,i2,j1,j2,0)] + f[IIJJK(g_nx-1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,2)] -= g_v[j1] / g_dx 
                    * f[IIJJK(g_nx-1,i2,j1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j1] / g_dx 
                    * f[IIJJK(g_nx-1,i2,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j1] / g_dx 
                    * f[IIJJK(g_nx-1,i2,j1,j2,4)] / 3.0;
            }
        }
        else if(g_upwindFlux && g_v[j1] <= 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j1] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,1)]);
            Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,1)]);
            Af[IIJJK(i1,i2,j1,j2,2)] -= g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j1] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(i1 < g_nx - 1) {
                Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j1] / g_dx 
                    * (f[IIJJK(i1+1,i2,j1,j2,0)] - f[IIJJK(i1+1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                    * (f[IIJJK(i1+1,i2,j1,j2,0)] - f[IIJJK(i1+1,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j1] / g_dx 
                    * f[IIJJK(i1+1,i2,j1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j1] / g_dx 
                    * f[IIJJK(i1+1,i2,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j1] / g_dx 
                    * f[IIJJK(i1+1,i2,j1,j2,4)] / 3.0;
            }
            
            if(i1 == g_nx-1 && g_isPeriodic) {
                Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j1] / g_dx 
                    * (f[IIJJK(0,i2,j1,j2,0)] - f[IIJJK(0,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j1] / g_dx 
                    * (f[IIJJK(0,i2,j1,j2,0)] - f[IIJJK(0,i2,j1,j2,1)]);
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j1] / g_dx 
                    * f[IIJJK(0,i2,j1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j1] / g_dx 
                    * f[IIJJK(0,i2,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j1] / g_dx 
                    * f[IIJJK(0,i2,j1,j2,4)] / 3.0;
            }
        }
        
        
        // Off diagonal i2-Term
        if(g_upwindFlux && g_v[j2] > 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,2)]);
            Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,2)]);
            Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(i2 > 0) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j2] / g_dx 
                    * (f[IIJJK(i1,i2-1,j1,j2,0)] + f[IIJJK(i1,i2-1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,1)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2-1,j1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,i2-1,j1,j2,0)] + f[IIJJK(i1,i2-1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2-1,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2-1,j1,j2,4)] / 3.0;
            }
            
            if(i2 == 0 && g_isPeriodic) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j2] / g_dx 
                    * (f[IIJJK(i1,g_nx-1,j1,j2,0)] + f[IIJJK(i1,g_nx-1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,1)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,g_nx-1,j1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,g_nx-1,j1,j2,0)] + f[IIJJK(i1,g_nx-1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,g_nx-1,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j2] / g_dx 
                    * f[IIJJK(i1,g_nx-1,j1,j2,4)] / 3.0;
            }
        }
        else if(g_upwindFlux && g_v[j2] <= 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] -= g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,2)]);
            Af[IIJJK(i1,i2,j1,j2,1)] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,2)]);
            Af[IIJJK(i1,i2,j1,j2,3)] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] -= g_v[j2] / g_dx 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(i2 < g_nx - 1) {
                Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,i2+1,j1,j2,0)] - f[IIJJK(i1,i2+1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2+1,j1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,i2+1,j1,j2,0)] - f[IIJJK(i1,i2+1,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2+1,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,i2+1,j1,j2,4)] / 3.0;
            }
            
            if(i2 == g_nx-1 && g_isPeriodic) {
                Af[IIJJK(i1,i2,j1,j2,0)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,0,j1,j2,0)] - f[IIJJK(i1,0,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,0,j1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += g_v[j2] / g_dx 
                    * (f[IIJJK(i1,0,j1,j2,0)] - f[IIJJK(i1,0,j1,j2,2)]);
                Af[IIJJK(i1,i2,j1,j2,3)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,0,j1,j2,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += g_v[j2] / g_dx 
                    * f[IIJJK(i1,0,j1,j2,4)] / 3.0;
            }
        }
        
        
        // Off diagonal j1-Term
        if(g_upwindFlux && s_E[NII(0,i1,i2)] > 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] += s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,3)]);
            Af[IIJJK(i1,i2,j1,j2,1)] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,3)]);
            Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(j1 > 0) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= s_E[NII(0,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1-1,j2,0)] + f[IIJJK(i1,i2,j1-1,j2,3)]);
                Af[IIJJK(i1,i2,j1,j2,1)] -= s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1-1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] -= s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1-1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(0,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1-1,j2,0)] + f[IIJJK(i1,i2,j1-1,j2,3)]);
                Af[IIJJK(i1,i2,j1,j2,4)] -= s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1-1,j2,4)] / 3.0;
            }
        }
        else if(g_upwindFlux && s_E[NII(0,i1,i2)] <= 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] -= s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,3)]);
            Af[IIJJK(i1,i2,j1,j2,1)] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(0,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,3)]);
            Af[IIJJK(i1,i2,j1,j2,4)] -= s_E[NII(0,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,4)] / 3.0;
            
            if(j1 < g_nv - 1) {
                Af[IIJJK(i1,i2,j1,j2,0)] += s_E[NII(0,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1+1,j2,0)] - f[IIJJK(i1,i2,j1+1,j2,3)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1+1,j2,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1+1,j2,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(0,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1+1,j2,0)] - f[IIJJK(i1,i2,j1+1,j2,3)]);
                Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(0,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1+1,j2,4)] / 3.0;
            }
        }
        
        
        // Off diagonal j2-Term
        if(g_upwindFlux && s_E[NII(1,i1,i2)] > 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] += s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,4)]);
            Af[IIJJK(i1,i2,j1,j2,1)] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] + f[IIJJK(i1,i2,j1,j2,4)]);
            
            if(j2 > 0) {
                Af[IIJJK(i1,i2,j1,j2,0)] -= s_E[NII(1,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1,j2-1,0)] + f[IIJJK(i1,i2,j1,j2-1,4)]);
                Af[IIJJK(i1,i2,j1,j2,1)] -= s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2-1,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] -= s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2-1,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] -= s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2-1,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(1,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1,j2-1,0)] + f[IIJJK(i1,i2,j1,j2-1,4)]);
            }
        }
        else if(g_upwindFlux && s_E[NII(1,i1,i2)] <= 0) {
            Af[IIJJK(i1,i2,j1,j2,0)] -= s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,4)]);
            Af[IIJJK(i1,i2,j1,j2,1)] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,1)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,2)] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,3)] -= s_E[NII(1,i1,i2)] / g_dv 
                * f[IIJJK(i1,i2,j1,j2,3)] / 3.0;
            Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(1,i1,i2)] / g_dv 
                * (f[IIJJK(i1,i2,j1,j2,0)] - f[IIJJK(i1,i2,j1,j2,4)]);
            
            if(j2 < g_nv - 1) {
                Af[IIJJK(i1,i2,j1,j2,0)] += s_E[NII(1,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1,j2+1,0)] - f[IIJJK(i1,i2,j1,j2+1,4)]);
                Af[IIJJK(i1,i2,j1,j2,1)] += s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2+1,1)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,2)] += s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2+1,2)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,3)] += s_E[NII(1,i1,i2)] / g_dv 
                    * f[IIJJK(i1,i2,j1,j2+1,3)] / 3.0;
                Af[IIJJK(i1,i2,j1,j2,4)] += s_E[NII(1,i1,i2)] / g_dv 
                    * (f[IIJJK(i1,i2,j1,j2+1,0)] - f[IIJJK(i1,i2,j1,j2+1,4)]);
            }
        }
    }}}}
    
    err = VecRestoreArrayRead(x, &f);CHKERRV(err);
    err = VecRestoreArray(y, &Af);CHKERRV(err);
}


// Do the sweep
void sweep2d(double sigma, double *E, double *q, double *f)
{
    int iter;
    PetscErrorCode err;
    double *temp;
    KSPConvergedReason reason;
    
    
    // Set static variables
    s_sigma = sigma;
    s_E = E;
    
    
    // Setup RHS
    err = VecGetArray(s_b, &temp);CHKERRV(err);
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        temp[IIJJK(i1,i2,j1,j2,0)] = q[IIJJK(i1,i2,j1,j2,0)];
        temp[IIJJK(i1,i2,j1,j2,1)] = q[IIJJK(i1,i2,j1,j2,1)] / 3.0;
        temp[IIJJK(i1,i2,j1,j2,2)] = q[IIJJK(i1,i2,j1,j2,2)] / 3.0;
        temp[IIJJK(i1,i2,j1,j2,3)] = q[IIJJK(i1,i2,j1,j2,3)] / 3.0;
        temp[IIJJK(i1,i2,j1,j2,4)] = q[IIJJK(i1,i2,j1,j2,4)] / 3.0;
    }}}}
    err = VecRestoreArray(s_b, &temp);CHKERRV(err);
    
    
    // GMRES
    err = KSPSolve(s_ksp, s_b, s_f);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp, &iter);CHKERRV(err);
    err = VecGetArray(s_f, &temp);CHKERRV(err);
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
    for(int k = 0; k < g_nBasis; k++) {
        f[IIJJK(i1,i2,j1,j2,k)] = temp[IIJJK(i1,i2,j1,j2,k)];
    }}}}}
    err = VecRestoreArray(s_f, &temp);CHKERRV(err);
    
    
    // Print and save sweep iters
    err = KSPGetConvergedReason(s_ksp, &reason);CHKERRV(err);
    if(g_printSweep)
        printf("   sweep2d:  GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}







