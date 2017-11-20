#include "global.h"
#include <math.h>
#include <stdio.h>


/*
    Manufactured solution to be used for convergence test.
*/
double conv_solution(double x, double v, double t)
{
    double e = exp(-(4.0 * v - 1.0) * (4.0 * v - 1.0) / 4.0);
    double c = cos(2.0 * x - 2.0 * M_PI * t);
    
    return (2.0 - c) * e;
}


/*
    E-field for convergence test.
*/
double conv_efield(double x, double t) 
{
    double s = sin(2.0 * x - 2.0 * M_PI * t);
    
    return -sqrt(M_PI) / 4.0 * s;
}


/*
    Source for convergence test.
*/
double conv_source(double x, double v, double t)
{
    double e = exp(-(4.0 * v - 1.0) * (4.0 * v - 1.0) / 4.0);
    double s = sin(2.0 * x - 2.0 * M_PI * t);
    double c = cos(2.0 * x - 2.0 * M_PI * t);
    
    return -2.0 * M_PI * s * e + 2.0 * v * s * e 
        + 0.5 * sqrt(M_PI) * s * (2.0-c) * e * (4.0*v-1.0);
}


/*
    L1 errors of f and E for convergence test.
*/
void conv_error(double *f, double *E, double t)
{
    double diffL1, normL1, diffL2, normL2, diffMax, normMax;
    
    
    // Convergence of f (uses 4th order Simpson's Rule)
    diffL1 = 0.0;
    normL1 = 0.0;
    diffL2 = 0.0;
    normL2 = 0.0;
    diffMax = 0.0;
    normMax = 0.0;
    for(int i = 0; i < g_nx; i++) {
    for(int j = 0; j < g_nv; j++) {
        double approx00 = f[IJK(i,j,0)] - f[IJK(i,j,1)] - f[IJK(i,j,2)];
        double approx01 = f[IJK(i,j,0)] - f[IJK(i,j,1)];
        double approx02 = f[IJK(i,j,0)] - f[IJK(i,j,1)] + f[IJK(i,j,2)];
        double approx10 = f[IJK(i,j,0)] - f[IJK(i,j,2)];
        double approx11 = f[IJK(i,j,0)];
        double approx12 = f[IJK(i,j,0)] + f[IJK(i,j,2)];
        double approx20 = f[IJK(i,j,0)] + f[IJK(i,j,1)] - f[IJK(i,j,2)];
        double approx21 = f[IJK(i,j,0)] + f[IJK(i,j,1)];
        double approx22 = f[IJK(i,j,0)] + f[IJK(i,j,1)] + f[IJK(i,j,2)];
        
        double actual00 = conv_solution(g_x[i]-g_dx/2, g_v[j]-g_dv/2, t);
        double actual01 = conv_solution(g_x[i]-g_dx/2, g_v[j], t);
        double actual02 = conv_solution(g_x[i]-g_dx/2, g_v[j]+g_dv/2, t);
        double actual10 = conv_solution(g_x[i], g_v[j]-g_dv/2, t);
        double actual11 = conv_solution(g_x[i], g_v[j], t);
        double actual12 = conv_solution(g_x[i], g_v[j]+g_dv/2, t);
        double actual20 = conv_solution(g_x[i]+g_dx/2, g_v[j]-g_dv/2, t);
        double actual21 = conv_solution(g_x[i]+g_dx/2, g_v[j], t);
        double actual22 = conv_solution(g_x[i]+g_dx/2, g_v[j]+g_dv/2, t);
        
        // L1 Difference
        diffL1 += g_dx * g_dv * fabs(approx00 - actual00) / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx01 - actual01) * 4.0 / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx02 - actual02) / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx10 - actual10) * 4.0 / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx11 - actual11) * 16.0 / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx12 - actual12) * 4.0 / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx20 - actual20) / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx21 - actual21) * 4.0 / 36.0;
        diffL1 += g_dx * g_dv * fabs(approx22 - actual22) / 36.0;
        
        normL1 += g_dx * g_dv * fabs(actual00) / 36.0;
        normL1 += g_dx * g_dv * fabs(actual01) * 4.0 / 36.0;
        normL1 += g_dx * g_dv * fabs(actual02) / 36.0;
        normL1 += g_dx * g_dv * fabs(actual10) * 4.0 / 36.0;
        normL1 += g_dx * g_dv * fabs(actual11) * 16.0 / 36.0;
        normL1 += g_dx * g_dv * fabs(actual12) * 4.0 / 36.0;
        normL1 += g_dx * g_dv * fabs(actual20) / 36.0;
        normL1 += g_dx * g_dv * fabs(actual21) * 4.0 / 36.0;
        normL1 += g_dx * g_dv * fabs(actual22) / 36.0;
        
        // L2 Difference
        diffL2 += g_dx * g_dv * pow(approx00 - actual00, 2) / 36.0;
        diffL2 += g_dx * g_dv * pow(approx01 - actual01, 2) * 4.0 / 36.0;
        diffL2 += g_dx * g_dv * pow(approx02 - actual02, 2) / 36.0;
        diffL2 += g_dx * g_dv * pow(approx10 - actual10, 2) * 4.0 / 36.0;
        diffL2 += g_dx * g_dv * pow(approx11 - actual11, 2) * 16.0 / 36.0;
        diffL2 += g_dx * g_dv * pow(approx12 - actual12, 2) * 4.0 / 36.0;
        diffL2 += g_dx * g_dv * pow(approx20 - actual20, 2) / 36.0;
        diffL2 += g_dx * g_dv * pow(approx21 - actual21, 2) * 4.0 / 36.0;
        diffL2 += g_dx * g_dv * pow(approx22 - actual22, 2) / 36.0;
        
        normL2 += g_dx * g_dv * pow(actual00, 2) / 36.0;
        normL2 += g_dx * g_dv * pow(actual01, 2) * 4.0 / 36.0;
        normL2 += g_dx * g_dv * pow(actual02, 2) / 36.0;
        normL2 += g_dx * g_dv * pow(actual10, 2) * 4.0 / 36.0;
        normL2 += g_dx * g_dv * pow(actual11, 2) * 16.0 / 36.0;
        normL2 += g_dx * g_dv * pow(actual12, 2) * 4.0 / 36.0;
        normL2 += g_dx * g_dv * pow(actual20, 2) / 36.0;
        normL2 += g_dx * g_dv * pow(actual21, 2) * 4.0 / 36.0;
        normL2 += g_dx * g_dv * pow(actual22, 2) / 36.0;
        
        // Max Difference
        if(fabs(approx00 - actual00) > diffMax)
            diffMax = fabs(approx00 - actual00);
        if(fabs(approx02 - actual02) > diffMax)
            diffMax = fabs(approx02 - actual02);
        if(fabs(approx20 - actual20) > diffMax)
            diffMax = fabs(approx20 - actual20);
        if(fabs(approx22 - actual22) > diffMax)
            diffMax = fabs(approx22 - actual22);
        
        if(fabs(actual00) > normMax)
            normMax = fabs(actual00);
        if(fabs(actual02) > normMax)
            normMax = fabs(actual02);
        if(fabs(actual20) > normMax)
            normMax = fabs(actual20);
        if(fabs(actual22) > normMax)
            normMax = fabs(actual22);
    }}
    
    printf("Convergence (L1,L2,Linf) rel err: (%e, %e, %e)\n", 
           diffL1 / normL1, sqrt(diffL2 / normL2), diffMax / normMax);
    
    
    // Convergence of E
    diffL1 = 0.0;
    normL1 = 0.0;
    diffL2 = 0.0;
    normL2 = 0.0;
    diffMax = 0.0;
    normMax = 0.0;
    for(int i = 0; i < g_nx; i++) {
        double approx = E[i];
        double actual = conv_efield(g_x[i], t);
        
        // L1
        diffL1 += g_dx * fabs(approx - actual);
        normL1 += g_dx * fabs(actual);
        
        // L2
        diffL2 += g_dx * pow(approx - actual, 2);
        normL2 += g_dx * pow(actual, 2);
        
        // Linf
        if(fabs(approx - actual) > diffMax)
            diffMax = fabs(approx - actual);
        if(fabs(actual) > normMax)
            normMax = fabs(actual);
    }
    
    printf("Convergence E-field (L1,L2,Linf) rel err: (%e, %e, %e)\n", 
           diffL1 / normL1, sqrt(diffL2 / normL2), diffMax / normMax);
}


/*
    Calculates rho from f.
    rho = <f>
*/
void calcRho(double *f, double *rho)
{
    // 1D
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx; i++) {
            rho[IK(i,0)] = rho[IK(i,1)] = 0.0;
            for(int j = 0; j < g_nv; j++) {
                rho[IK(i,0)] += g_dv * f[IJK(i,j,0)];
                rho[IK(i,1)] += g_dv * f[IJK(i,j,1)];
            }
        }
    }
    // 2D
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            rho[IIK(i1,i2,0)] = rho[IIK(i1,i2,1)] = rho[IIK(i1,i2,2)] = 0.0;
            for(int j1 = 0; j1 < g_nv; j1++) {
            for(int j2 = 0; j2 < g_nv; j2++) {
                rho[IIK(i1,i2,0)] += g_dv * g_dv * f[IIJJK(i1,i2,j1,j2,0)];
                rho[IIK(i1,i2,1)] += g_dv * g_dv * f[IIJJK(i1,i2,j1,j2,1)];
                rho[IIK(i1,i2,2)] += g_dv * g_dv * f[IIJJK(i1,i2,j1,j2,2)];
            }}
        }}
    }
}


/*
    Solve a small NxN linear system.
*/
void solveNxN(int n, double *A_, double *b)
{
    #define A(i,j) A_[((i)*n+(j))]
    
    
    // Triangularize system
    for(int pivot = 0; pivot < n; pivot++) {
        
        // Normalize A(pivot,pivot)
        b[pivot] = b[pivot] / A(pivot,pivot);
        for(int j = n-1; j > pivot; j--) {
            A(pivot,j) = A(pivot,j) / A(pivot,pivot);
        }
        //A(pivot,pivot) = 1.0;
        
        // Zero A(i,pivot)
        for(int i = pivot + 1; i < n; i++) {
            b[i] = b[i] - A(i,pivot) * b[pivot];
            for(int j = n-1; j > pivot; j--) {
                A(i,j) = A(i,j) - A(i,pivot) * A(pivot,j);
            }
            //A(i,pivot) = 0.0;
        }
    }
    
    
    // Backward solve
    for(int i = n-2; i >= 0; i--) {
    for(int j = i+1; j < n; j++) {
        b[i] = b[i] - A(i,j) * b[j];
    }}
    
    #undef A
}


