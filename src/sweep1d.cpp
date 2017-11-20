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
    const double *temp;
    
    static bool firstTime = true;
    static double *q = NULL;
    if(firstTime) {
        int n = g_nx * g_nv * g_nBasis;
        
        firstTime = false;
        q = new double[n];
    }
    
    VecCopy(x, y);
    err = VecGetArrayRead(x, &temp);CHKERRQ(err);
    err = VecGetArray(y, &f);CHKERRQ(err);
    
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        q[IJK(i,j,0)] = temp[IJK(i,j,0)];
        q[IJK(i,j,1)] = temp[IJK(i,j,1)] * 3.0;
        q[IJK(i,j,2)] = temp[IJK(i,j,2)] * 3.0;
    }}
    sweep1dFast(s_sigma, s_E, q, f);
    
    err = VecRestoreArray(y, &f);CHKERRQ(err);
    err = VecRestoreArrayRead(x, &temp);CHKERRQ(err);
    
    return err;
}


/*
    Allocate static variables.
*/
void sweep1d_init()
{
    int n = g_nx * g_nv * g_nBasis;
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


/*
    Free static variables.
*/
void sweep1d_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_b);CHKERRV(err);
    err = VecDestroy(&s_f);CHKERRV(err);
    err = MatDestroy(&s_A);CHKERRV(err);
    err = KSPDestroy(&s_ksp);CHKERRV(err);
}



/*
    Matrix to apply for GMRES
*/
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
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        
        // Diagonal Term
        Af[IJK(i,j,0)] = s_sigma * f[IJK(i,j,0)];
        Af[IJK(i,j,1)] = s_sigma / 3.0 * f[IJK(i,j,1)];
        Af[IJK(i,j,2)] = s_sigma / 3.0 * f[IJK(i,j,2)];
        
        Af[IJK(i,j,1)] -= 2.0 * g_v[j] / g_dx * f[IJK(i,j,0)];
        Af[IJK(i,j,2)] -= 2.0 * s_E[i] / g_dv * f[IJK(i,j,0)];
        
        
        // Off diagonal i-Term
        if(g_upwindFlux && g_v[j] > 0) {
            Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i,j,0)] + f[IJK(i,j,1)]);
            Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i,j,0)] + f[IJK(i,j,1)]);
            Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i,j,2)] / 3.0;
            
            if(i > 0) {
                Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
                Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i-1,j,2)] / 3.0;
            }
            
            if(i == 0 && g_isPeriodic) {
                Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]);
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]);
                Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(g_nx-1,j,2)] / 3.0;
            }
        }
        else if(g_upwindFlux && g_v[j] <= 0) {
            Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i,j,0)] - f[IJK(i,j,1)]);
            Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i,j,0)] - f[IJK(i,j,1)]);
            Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i,j,2)] / 3.0;
            
            if(i < g_nx - 1) {
                Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
                Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i+1,j,2)] / 3.0;
            }
            
            if(i == g_nx - 1 && g_isPeriodic) {
                Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]);
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]);
                Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(0,j,2)] / 3.0;
            }
        }
        else if(!g_upwindFlux) {
            Af[IJK(i,j,0)] += g_v[j] / g_dx * f[IJK(i,j,1)];
            Af[IJK(i,j,1)] += g_v[j] / g_dx * f[IJK(i,j,0)];
            
            if(i < g_nx - 1) {
                Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]) / 2.0;
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]) / 2.0;
                Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i+1,j,2)] / 6.0;
            }
            
            if(i == g_nx - 1 && g_isPeriodic) {
                Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]) / 2.0;
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]) / 2.0;
                Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(0,j,2)] / 6.0;
            }
            
            if(i > 0) {
                Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]) / 2.0;
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]) / 2.0;
                Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i-1,j,2)] / 6.0;
            }
            
            if(i == 0 && g_isPeriodic) {
                Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]) / 2.0;
                Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]) / 2.0;
                Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(g_nx-1,j,2)] / 6.0;
            }
        }
        
        
        // Off diagonal j-Term
        if(g_upwindFlux && s_E[i] > 0) {
            Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j,0)] + f[IJK(i,j,2)]);
            Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j,1)] / 3.0;
            Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j,0)] + f[IJK(i,j,2)]);
            
            if(j > 0) {
                Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
                Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j-1,1)] / 3.0;
                Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
            }
        }
        else if(g_upwindFlux && s_E[i] <= 0) {
            Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j,0)] - f[IJK(i,j,2)]);
            Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j,1)] / 3.0;
            Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j,0)] - f[IJK(i,j,2)]);
            
            if(j < g_nv - 1) {
                Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
                Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j+1,1)] / 3.0;
                Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
            }
        }
        else if(!g_upwindFlux) {
            Af[IJK(i,j,0)] += s_E[i] / g_dv * f[IJK(i,j,2)];
            Af[IJK(i,j,2)] += s_E[i] / g_dv * f[IJK(i,j,0)];
            
            if(j < g_nv - 1) {
                Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]) / 2.0;
                Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j+1,1)] / 6.0;
                Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]) / 2.0;
            }
            
            if(j > 0) {
                Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]) / 2.0;
                Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j-1,1)] / 6.0;
                Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]) / 2.0;
            }
        }
    }}
    
    err = VecRestoreArrayRead(x, &f);CHKERRV(err);
    err = VecRestoreArray(y, &Af);CHKERRV(err);
}


/*
    Do the sweep
*/
void sweep1d(double sigma, double *E, double *q, double *f)
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
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        temp[IJK(i,j,0)] = q[IJK(i,j,0)];
        temp[IJK(i,j,1)] = q[IJK(i,j,1)] / 3.0;
        temp[IJK(i,j,2)] = q[IJK(i,j,2)] / 3.0;
    }}
    err = VecRestoreArray(s_b, &temp);CHKERRV(err);
    
    
    // GMRES
    err = KSPSolve(s_ksp, s_b, s_f);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp, &iter);CHKERRV(err);
    err = VecGetArray(s_f, &temp);CHKERRV(err);
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
    for(int k = 0; k < g_nBasis; k++) {
        f[IJK(i,j,k)] = temp[IJK(i,j,k)];
    }}}
    err = VecRestoreArray(s_f, &temp);CHKERRV(err);
    
    
    // Print and save sweep iters
    err = KSPGetConvergedReason(s_ksp, &reason);CHKERRV(err);
    if(g_printSweep)
        printf("   sweep1d:  GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}







