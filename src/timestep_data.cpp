#include "global.h"
#include <stdio.h>
#include <math.h>


/*
    Return max value.
*/
static inline
double max(double a, double b)
{
    if(a > b)
        return a;
    return b;
}


/*
    Calculates the l2 norm of the efield for the average over 
    x-dimension 1 and 2.
*/
static
void calcEfieldL2(double *E, double *l2norm1, double *l2norm2)
{
    double l21 = 0.0;
    double l22 = 0.0;
    
    for(int i1 = 0; i1 < g_nx; i1++) {
        double Eavg = 0.0;
        for(int i2 = 0; i2 < g_nx; i2++) {
            Eavg += g_dx * E[NII(0,i1,i2)];
        }
        Eavg = Eavg / (g_bx - g_ax);
        l21 += g_dx * Eavg * Eavg;
    }
    
    for(int i2 = 0; i2 < g_nx; i2++) {
        double Eavg = 0.0;
        for(int i1 = 0; i1 < g_nx; i1++) {
            Eavg += g_dx * E[NII(1,i1,i2)];
        }
        Eavg = Eavg / (g_bx - g_ax);
        l22 += g_dx * Eavg * Eavg;
    }
    
    *l2norm1 = sqrt(l21);
    *l2norm2 = sqrt(l22);
}


/*
    Calculate min, max, mean of e-field.
*/
static
void calcMinMaxEfield(double *E, double *min, double *max, double *mean)
{
    *min = fabs(E[0]);
    *max = fabs(E[0]);
    *mean = 0.0;
    for(size_t i = 0; i < g_ESize; i++) {
        double Ei = fabs(E[i]);
        
        // max
        if(Ei > *max)
            *max = Ei;
        
        // min
        if(Ei < *min)
            *min = Ei;
        
        // mean
        *mean += Ei;
    }
    
    // Finish mean calculation
    *mean = *mean / g_ESize;
}


/*
    Calculate mass, momentum, energy, l2norm(f), and l2norm(E).
    Puts these into time step data.
    Prints results.
*/
void calcTimeStepQuantities1D(double dt, double t, double *f, double *E)
{
    double mass, momentum, energy, l2norm, l2efield;
    double maxEfield, minEfield, meanEfield;
    TimeStepData tsd;
    
    
    // Initialize
    mass = 0.0;
    momentum = 0.0;
    energy = 0.0;
    l2norm = 0.0;
    l2efield = 0.0;
    
    
    // Conserved quantities
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        mass += f[IJK(i,j,0)];
        momentum += g_v[j] * f[IJK(i,j,0)] + g_dv / 6.0 * f[IJK(i,j,2)];
        energy += g_v[j] * g_v[j] * f[IJK(i,j,0)] / 2.0;
        l2norm += f[IJK(i,j,0)] * f[IJK(i,j,0)]
                + f[IJK(i,j,1)] * f[IJK(i,j,1)] / 3.0
                + f[IJK(i,j,2)] * f[IJK(i,j,2)] / 3.0;
    }}
    
    mass     = mass     * g_dx * g_dv;
    momentum = momentum * g_dx * g_dv;
    energy   = energy   * g_dx * g_dv;
    l2norm   = l2norm   * g_dx * g_dv;
    
    for(int i = 0; i < g_nx; i++) {
        energy += E[i] * E[i] / 2.0 * g_dx;
    }
    
    
    // L^2 of Efield
    for(int i = 0; i < g_nx; i++) {
        l2efield += E[i] * E[i] * g_dx;
    }
    l2efield = sqrt(l2efield);
    
    
    // Calc min, max, mean of e-field
    calcMinMaxEfield(E, &minEfield, &maxEfield, &meanEfield);
    
    
    // Store data
    tsd.time = t;
    tsd.l2efield1 = l2efield;
    tsd.l2efield2 = 0.0;
    tsd.mass = mass;
    tsd.momentum1 = momentum;
    tsd.momentum2 = 0.0;
    tsd.energy = energy;
    tsd.l2norm = l2norm;
    tsd.minECFL = dt * minEfield / g_dv;
    tsd.maxECFL = dt * maxEfield / g_dv;
    tsd.meanECFL = dt * meanEfield / g_dv;
    tsd.xCFL = dt * max(fabs(g_x[0]), fabs(g_x[g_nx-1])) / g_dx;
    g_timeStepData.push_back(tsd);
    
    
    // Print conserved quantities
    if(g_printConserved) {
        printf("   mass = %.2e, mom = %.2e, energy = %.2e, l2norm = %.2e\n", 
               mass, momentum, energy, l2norm);
    }
    if(g_printCFL) {
        printf("   minECFL = %.2e, maxECFL = %.2e, meanECFL = %.2e, xCFL = %.2e\n", 
               tsd.minECFL, tsd.maxECFL, tsd.meanECFL, tsd.xCFL);
    }
}


/*
    Calculate mass, momentum, energy, l2norm(f), and l2norm(E).
    Puts these into time step data.
    Prints results.
*/
void calcTimeStepQuantities2D(double dt, double t, double *f, double *E)
{
    double mass, momentum1, momentum2, energy, l2norm, l2efield1, l2efield2;
    double maxEfield, minEfield, meanEfield;
    TimeStepData tsd;
    
    
    // Initialize
    mass = 0.0;
    momentum1 = 0.0;
    momentum2 = 0.0;
    energy = 0.0;
    l2norm = 0.0;
    
    
    // Conserved quantities
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
    for(int j1 = 0; j1 < g_nv; j1++) {
    for(int j2 = 0; j2 < g_nv; j2++) {
        mass += f[IIJJK(i1,i2,j1,j2,0)];
        momentum1 += g_v[j1] * f[IIJJK(i1,i2,j1,j2,0)];
        momentum2 += g_v[j2] * f[IIJJK(i1,i2,j1,j2,0)];
        energy += (g_v[j1] * g_v[j1] + g_v[j2] * g_v[j2]) 
                * f[IIJJK(i1,i2,j1,j2,0)] / 2.0;
        l2norm += f[IIJJK(i1,i2,j1,j2,0)] * f[IIJJK(i1,i2,j1,j2,0)]
                + f[IIJJK(i1,i2,j1,j2,1)] * f[IIJJK(i1,i2,j1,j2,1)] / 3.0
                + f[IIJJK(i1,i2,j1,j2,2)] * f[IIJJK(i1,i2,j1,j2,2)] / 3.0;
    }}}}
    
    mass      = mass      * g_dx * g_dx * g_dv * g_dv;
    momentum1 = momentum1 * g_dx * g_dx * g_dv * g_dv;
    momentum2 = momentum2 * g_dx * g_dx * g_dv * g_dv;
    energy    = energy    * g_dx * g_dx * g_dv * g_dv;
    l2norm    = l2norm    * g_dx * g_dx * g_dv * g_dv;
    
    for(int i1 = 0; i1 < g_nx; i1++) {
    for(int i2 = 0; i2 < g_nx; i2++) {
        energy += (E[NII(0,i1,i2)] * E[NII(0,i1,i2)] 
                   + E[NII(1,i1,i2)] * E[NII(1,i1,i2)]) / 2.0 * g_dx * g_dx;
    }}
    
    
    // L^2 of Efield
    calcEfieldL2(E, &l2efield1, &l2efield2);
    
    
    // Calc min, max, mean of e-field
    calcMinMaxEfield(E, &minEfield, &maxEfield, &meanEfield);
    
    
    // Store data
    tsd.time = t;
    tsd.l2efield1 = l2efield1;
    tsd.l2efield2 = l2efield2;
    tsd.mass = mass;
    tsd.momentum1 = momentum1;
    tsd.momentum2 = momentum2;
    tsd.energy = energy;
    tsd.l2norm = l2norm;
    tsd.minECFL = dt * minEfield / g_dv;
    tsd.maxECFL = dt * maxEfield / g_dv;
    tsd.meanECFL = dt * meanEfield / g_dv;
    tsd.xCFL = dt * max(fabs(g_x[0]), fabs(g_x[g_nx-1])) / g_dx;
    g_timeStepData.push_back(tsd);
    
    
    // Print conserved quantities
    if(g_printConserved) {
        printf("   mass = %.2e, mom1 = %.2e, mom2 = %.2e, energy = %.2e, l2norm = %.2e\n", 
               mass, momentum1, momentum2, energy, l2norm);
    }
}


