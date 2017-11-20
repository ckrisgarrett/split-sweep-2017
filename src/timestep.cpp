#include "global.h"
#include <math.h>
#include <string.h>
#include <stdio.h>


// Memory to be allocated at program begin.
static double *s_q, *s_f1, *s_f2;


/*
    Source necessary for convergence test.
*/
static 
void addConvergenceSource(double t, double *q)
{
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        double x1, x2, v1, v2;
        double q11, q12, q21, q22;
        
        x1 = g_x[i] - g_dx / 2.0;
        x2 = g_x[i] + g_dx / 2.0;
        v1 = g_v[j] - g_dv / 2.0;
        v2 = g_v[j] + g_dv / 2.0;
        
        q11 = conv_source(x1, v1, t);
        q12 = conv_source(x1, v2, t);
        q21 = conv_source(x2, v1, t);
        q22 = conv_source(x2, v2, t);
        
        q[IJK(i,j,0)] += 0.25 * (q22 + q21 + q12 + q11);
        q[IJK(i,j,1)] += 0.25 * (q22 + q21 - q12 - q11);
        q[IJK(i,j,2)] += 0.25 * (q22 - q21 + q12 - q11);
    }}
}


/*
    Backward Euler scheme
*/
static
void timestep1(double t, double dt, double *E, double *f)
{
    double *q = s_q;
    double sigma = 1.0 / dt;
    
    
    // Set source
    for(size_t index = 0; index < g_fSize; index++) {
        q[index] = 1.0 / dt * f[index];
    }
    if(g_runType == RUN_CONVERGENCE) {
        addConvergenceSource(t + dt, q);
    }
    
    
    // Do Euler step
    if(g_useNonlinear)
        eulerStepNonlinear(sigma, E, q, f);
    else 
        eulerStep(t, dt, sigma, E, q, f);
}


/*
    Second order time integrator.  L-Stable SDIRK Method.
    
    Butcher Table
     gamma  |   gamma     0
       1    |  1-gamma  gamma
    -------------------------
            |  1-gamma  gamma
*/
static
void timestep2(double t, double dt, double *E, double *f)
{
    double *q, *f1, *f2, *Af1, *Af2;
    const double gamma = 1.0 - 1.0 / sqrt(2.0);
    double sigma;
    
    
    // Memory necessary for computations
    q   = s_q;
    f1  = s_f1;
    f2  = s_f2;
    Af1 = s_f1;
    Af2 = s_f2;
    
    
    ///////// Step 1 /////////
    
    // Set sigma
    sigma = 1.0 / (gamma * dt);
    
    
    // Set source
    for(size_t index = 0; index < g_fSize; index++) {
        q[index] = sigma * f[index];
    }
    if(g_runType == RUN_CONVERGENCE) {
        addConvergenceSource(t + gamma * dt, q);
    }
    
    
    // Save f and do Euler step
    memcpy(f1, f, g_fSize * sizeof(double));
    if(g_useNonlinear)
        eulerStepNonlinear(sigma, E, q, f1);
    else 
        eulerStep(t, gamma * dt, sigma, E, q, f1);
    
    
    // First stage
    for(size_t index = 0; index < g_fSize; index++) {
        Af1[index] = sigma * (f[index] - f1[index]);
    }
    
    
    ///////// Step 2 /////////
    
    // Set sigma
    sigma = 1.0 / (gamma * dt);
    
    
    // Set source
    for(size_t index = 0; index < g_fSize; index++) {
        q[index] = sigma * f[index]
                 - sigma * (1.0-gamma) * dt * Af1[index];
    }
    if(g_runType == RUN_CONVERGENCE) {
        addConvergenceSource(t + dt, q);
    }
    
    
    // Save f and do Euler step
    memcpy(f2, f, g_fSize * sizeof(double));
    if(g_useNonlinear)
        eulerStepNonlinear(sigma, E, q, f2);
    else 
        eulerStep(t, dt, sigma, E, q, f2);
    
    
    // Second stage
    for(size_t index = 0; index < g_fSize; index++) {
        Af2[index] = sigma * (f[index] - f2[index]) 
                   - sigma * (1.0-gamma) * dt * Af1[index];
    }
    
    
    ///////// Put Stages Together /////////
    
    for(size_t index = 0; index < g_fSize; index++) {
        f[index] -= dt * ((1.0 - gamma) * Af1[index] + gamma * Af2[index]);
    }
}


/*
    Initialize memory.
*/
void timestep_init()
{
    s_q = new double[g_fSize];
    if(g_timeOrder == 2) {
        s_f1 = new double[g_fSize];
        s_f2 = new double[g_fSize];
    }
}


/*
    Free memory.
*/
void timestep_end()
{
    delete[] s_q;
    if(g_timeOrder == 2) {
        delete[] s_f1;
        delete[] s_f2;
    }
}


/*
    Select timestep from order.
*/
void timestep(double t, double dt, double *E, double *f)
{
    if(g_timeOrder == 1) 
        timestep1(t, dt, E, f);
    else if(g_timeOrder == 2)
        timestep2(t, dt, E, f);
    else 
        printf("Time Order out of bounds.\n");
}




