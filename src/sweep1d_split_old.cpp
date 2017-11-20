/*
    Solves:
    v dx f + sigma f = q    or
    E dv f + sigma f = q
    
    I hacked sweep.cpp to create this file.  It is not well optimized.
    Use at your own risk!!!!!
*/


#include "global.h"
#include <petscksp.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>


// Variables needed for this file.
static KSP s_ksp_v;
static KSP s_ksp_E;
static Mat s_A_v;
static Mat s_A_E;
static Vec s_b, s_f;
static double s_sigma, *s_E;

static
void mult_v(Mat A, Vec x, Vec y);

static
void mult_E(Mat A, Vec x, Vec y);


/*
    Allocate static variables.
*/
void sweep1dSplit_init()
{
    int n = g_nx * g_nv * g_nBasis;
    PetscErrorCode err;
    
    // Petsc init vectors
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_b);CHKERRV(err);
    err = VecCreateSeq(PETSC_COMM_SELF, n, &s_f);CHKERRV(err);
    
    // Petsc init (v)
    err = MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, NULL, &s_A_v);CHKERRV(err);
    err = MatShellSetOperation(s_A_v, MATOP_MULT, (void (*)(void))mult_v);CHKERRV(err);
    
    err = KSPCreate(PETSC_COMM_WORLD, &s_ksp_v);CHKERRV(err);
    err = KSPSetOperators(s_ksp_v,s_A_v,s_A_v);CHKERRV(err);
    
    err = KSPSetTolerances(s_ksp_v, g_sweepTol, PETSC_DEFAULT, PETSC_DEFAULT, 
                           g_sweepMaxiter);CHKERRV(err);
    err = KSPSetInitialGuessNonzero(s_ksp_v, PETSC_TRUE);CHKERRV(err);
    err = KSPSetType(s_ksp_v, KSPGMRES);CHKERRV(err);
    
    // Petsc init (E)
    err = MatCreateShell(PETSC_COMM_WORLD, n, n, n, n, NULL, &s_A_E);CHKERRV(err);
    err = MatShellSetOperation(s_A_E, MATOP_MULT, (void (*)(void))mult_E);CHKERRV(err);
    
    err = KSPCreate(PETSC_COMM_WORLD, &s_ksp_E);CHKERRV(err);
    err = KSPSetOperators(s_ksp_E,s_A_E,s_A_E);CHKERRV(err);
    
    err = KSPSetTolerances(s_ksp_E, g_sweepTol, PETSC_DEFAULT, PETSC_DEFAULT, 
                           g_sweepMaxiter);CHKERRV(err);
    err = KSPSetInitialGuessNonzero(s_ksp_E, PETSC_TRUE);CHKERRV(err);
    err = KSPSetType(s_ksp_E, KSPGMRES);CHKERRV(err);
}


/*
    Free static variables.
*/
void sweep1dSplit_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_b);CHKERRV(err);
    err = VecDestroy(&s_f);CHKERRV(err);
    err = MatDestroy(&s_A_v);CHKERRV(err);
    err = MatDestroy(&s_A_E);CHKERRV(err);
    err = KSPDestroy(&s_ksp_v);CHKERRV(err);
    err = KSPDestroy(&s_ksp_E);CHKERRV(err);
}



/*
    Matrix to apply for GMRES
*/
static
void mult_v(Mat A, Vec x, Vec y)
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
        //Af[IJK(i,j,2)] -= 2.0 * s_E[i] / g_dv * f[IJK(i,j,0)];
        
        
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
        // if(g_upwindFlux && s_E[i] > 0) {
//             Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j,0)] + f[IJK(i,j,2)]);
//             Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j,1)] / 3.0;
//             Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j,0)] + f[IJK(i,j,2)]);
//             
//             if(j > 0) {
//                 Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
//                 Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j-1,1)] / 3.0;
//                 Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]);
//             }
//         }
//         else if(g_upwindFlux && s_E[i] <= 0) {
//             Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j,0)] - f[IJK(i,j,2)]);
//             Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j,1)] / 3.0;
//             Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j,0)] - f[IJK(i,j,2)]);
//             
//             if(j < g_nv - 1) {
//                 Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
//                 Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j+1,1)] / 3.0;
//                 Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]);
//             }
//         }
//         else if(!g_upwindFlux) {
//             Af[IJK(i,j,0)] += s_E[i] / g_dv * f[IJK(i,j,2)];
//             Af[IJK(i,j,2)] += s_E[i] / g_dv * f[IJK(i,j,0)];
//             
//             if(j < g_nv - 1) {
//                 Af[IJK(i,j,0)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]) / 2.0;
//                 Af[IJK(i,j,1)] += s_E[i] / g_dv * f[IJK(i,j+1,1)] / 6.0;
//                 Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j+1,0)] - f[IJK(i,j+1,2)]) / 2.0;
//             }
//             
//             if(j > 0) {
//                 Af[IJK(i,j,0)] -= s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]) / 2.0;
//                 Af[IJK(i,j,1)] -= s_E[i] / g_dv * f[IJK(i,j-1,1)] / 6.0;
//                 Af[IJK(i,j,2)] += s_E[i] / g_dv * (f[IJK(i,j-1,0)] + f[IJK(i,j-1,2)]) / 2.0;
//             }
//         }
    }}
    
    err = VecRestoreArrayRead(x, &f);CHKERRV(err);
    err = VecRestoreArray(y, &Af);CHKERRV(err);
}


static
void mult_E(Mat A, Vec x, Vec y)
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
        
        //Af[IJK(i,j,1)] -= 2.0 * g_v[j] / g_dx * f[IJK(i,j,0)];
        Af[IJK(i,j,2)] -= 2.0 * s_E[i] / g_dv * f[IJK(i,j,0)];
        
        
        // Off diagonal i-Term
        // if(g_upwindFlux && g_v[j] > 0) {
//             Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i,j,0)] + f[IJK(i,j,1)]);
//             Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i,j,0)] + f[IJK(i,j,1)]);
//             Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i,j,2)] / 3.0;
//             
//             if(i > 0) {
//                 Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]);
//                 Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i-1,j,2)] / 3.0;
//             }
//             
//             if(i == 0 && g_isPeriodic) {
//                 Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]);
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]);
//                 Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(g_nx-1,j,2)] / 3.0;
//             }
//         }
//         else if(g_upwindFlux && g_v[j] <= 0) {
//             Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i,j,0)] - f[IJK(i,j,1)]);
//             Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i,j,0)] - f[IJK(i,j,1)]);
//             Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i,j,2)] / 3.0;
//             
//             if(i < g_nx - 1) {
//                 Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]);
//                 Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i+1,j,2)] / 3.0;
//             }
//             
//             if(i == g_nx - 1 && g_isPeriodic) {
//                 Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]);
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]);
//                 Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(0,j,2)] / 3.0;
//             }
//         }
//         else if(!g_upwindFlux) {
//             Af[IJK(i,j,0)] += g_v[j] / g_dx * f[IJK(i,j,1)];
//             Af[IJK(i,j,1)] += g_v[j] / g_dx * f[IJK(i,j,0)];
//             
//             if(i < g_nx - 1) {
//                 Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]) / 2.0;
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i+1,j,0)] - f[IJK(i+1,j,1)]) / 2.0;
//                 Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(i+1,j,2)] / 6.0;
//             }
//             
//             if(i == g_nx - 1 && g_isPeriodic) {
//                 Af[IJK(i,j,0)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]) / 2.0;
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(0,j,0)] - f[IJK(0,j,1)]) / 2.0;
//                 Af[IJK(i,j,2)] += g_v[j] / g_dx * f[IJK(0,j,2)] / 6.0;
//             }
//             
//             if(i > 0) {
//                 Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]) / 2.0;
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(i-1,j,0)] + f[IJK(i-1,j,1)]) / 2.0;
//                 Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(i-1,j,2)] / 6.0;
//             }
//             
//             if(i == 0 && g_isPeriodic) {
//                 Af[IJK(i,j,0)] -= g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]) / 2.0;
//                 Af[IJK(i,j,1)] += g_v[j] / g_dx * (f[IJK(g_nx-1,j,0)] + f[IJK(g_nx-1,j,1)]) / 2.0;
//                 Af[IJK(i,j,2)] -= g_v[j] / g_dx * f[IJK(g_nx-1,j,2)] / 6.0;
//             }
//         }
        
        
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
    Do the sweep (v)
*/
void sweep1dSplit_v(double sigma, double *q, double *f)
{
    int iter;
    PetscErrorCode err;
    double *temp;
    KSPConvergedReason reason;
    
    
    // Set static variables
    s_sigma = sigma;
    s_E = NULL;
    
    
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
    err = KSPSolve(s_ksp_v, s_b, s_f);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp_v, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp_v, &iter);CHKERRV(err);
    err = VecGetArray(s_f, &temp);CHKERRV(err);
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
    for(int k = 0; k < g_nBasis; k++) {
        f[IJK(i,j,k)] = temp[IJK(i,j,k)];
    }}}
    err = VecRestoreArray(s_f, &temp);CHKERRV(err);
    
    
    // Print and save sweep iters
    err = KSPGetConvergedReason(s_ksp_v, &reason);CHKERRV(err);
    if(g_printSweep)
        printf("   sweep1d:  GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}


/*
    Do the sweep (E)
*/
void sweep1dSplit_E(double sigma, double *E, double *q, double *f)
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
    err = KSPSolve(s_ksp_E, s_b, s_f);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp_E, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp_E, &iter);CHKERRV(err);
    err = VecGetArray(s_f, &temp);CHKERRV(err);
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
    for(int k = 0; k < g_nBasis; k++) {
        f[IJK(i,j,k)] = temp[IJK(i,j,k)];
    }}}
    err = VecRestoreArray(s_f, &temp);CHKERRV(err);
    
    
    // Print and save sweep iters
    err = KSPGetConvergedReason(s_ksp_E, &reason);CHKERRV(err);
    if(g_printSweep)
        printf("   sweep1d:  GMRES stats... iter: %d   reason: %s\n", 
               iter, KSPConvergedReasons[reason]);
    g_sweepIters.push_back(iter);
}




