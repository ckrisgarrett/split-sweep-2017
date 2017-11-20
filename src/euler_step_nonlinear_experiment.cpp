#include "global.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


static double *s_Evecs, *s_Fvecs, *s_E, *s_G, *s_alpha; // For eulerStepNon
static double *s_rho;                                   // For Gfunc
static double *s_A, *s_b, *s_work;                      // For leastSquares


/*
    LAPACK command to solve least squares problem.
*/
extern "C"
void dgels_(char *trans, int *rows, int *cols, int *nrhs, double *A, int *lda, 
            double *b, int *ldb, double *work, int *lwork, int *info);


/*
    Index for matrix with g_ESize rows.
    This is column major format.
*/
inline static
int VECS(int i, int j) {
    return i + j * g_ESize;
}


/*
    Allocate memory
*/
void eulerStepNonlinear_init()
{
    // For eulerStepNonlinear
    s_Evecs = new double[g_ESize * (g_eulerRestart + 1)];
    s_Fvecs = new double[g_ESize * (g_eulerRestart + 1)];
    s_E = new double[g_ESize];
    s_G = new double[g_ESize];
    s_alpha = new double[g_eulerRestart+1];
    
    // For Gfunc
    s_rho = new double[g_rhoSize];
    
    // For leastSquares
    s_A = new double[g_ESize * g_eulerRestart];
    s_b = new double[g_ESize];
    s_work = new double[g_eulerRestart + g_ESize];
}


/*
    Free memory
*/
void eulerStepNonlinear_end()
{
    delete[] s_Evecs;
    delete[] s_Fvecs;
    delete[] s_E;
    delete[] s_G;
    delete[] s_alpha;
    delete[] s_rho;
    delete[] s_A;
    delete[] s_b;
    delete[] s_work;
}


/*
    If vecs = (v_0, ..., v_m), this transforms vecs into
    (v_1, ..., v_m, 0)
*/
static
void shift(int m, double *vecs)
{
    for(int j = 0; j < m; j++) {
    for(size_t i = 0; i < g_ESize; i++) {
        vecs[VECS(i,j)] = vecs[VECS(i,j+1)];
    }}
    
    for(size_t i = 0; i < g_ESize; i++) {
        vecs[VECS(i,m)] = 0.0;
    }
}


/*
    G is the function where we want G(E) = E.
    This function computes G(E0) = E1.
*/
static
void Gfunc(double sigma, double *E0, double *q, double *f, double *E1, bool firstCall = false)
{
    static bool firstTime = true;
    static double *q1 = NULL;
    static double *qInit = NULL;
    static double *fInit = NULL;
    if(firstTime) {
        q1    = new double[g_fSize];
        qInit = new double[g_fSize];
        fInit = new double[g_fSize];
        firstTime = false;
    }
    
    if(firstCall) {
        for(size_t index = 0; index < g_fSize; index++) {
            qInit[index] = q[index] - sigma * f[index];
            fInit[index] = f[index];
        }
    }
    
    // Sweep to get f
    if(g_dimension == 1) {
        if(g_useNewSweep)
            sweep1dFast(sigma, E0, q, f);
        else {
            //sweep1d(sigma, E0, q, f);
            for(size_t index = 0; index < g_fSize; index++) {
                q1[index] = 2.0 * sigma * fInit[index];
            }
            sweep1dSplit_v(sigma * 2.0, q1, f);
            
            for(size_t index = 0; index < g_fSize; index++) {
                q1[index] = 1.0 * sigma * f[index] + qInit[index];
            }
            sweep1dSplit_E(sigma, E0, q1, f);
            
            for(size_t index = 0; index < g_fSize; index++) {
                q1[index] = 2.0 * sigma * f[index];
            }
            sweep1dSplit_v(sigma * 2.0, q1, f);
        }
    }
    else {
        if(g_useNewSweep)
            sweep2dFast(sigma, E0, q, f);
        else
            sweep2d(sigma, E0, q, f);
    }
    
    
    // Get rho
    calcRho(f, s_rho);
    
    
    // Calculate E1
    efield(s_rho, E1);
}


/*
    Calculates min ||Fvecs * alpha||_2 such that sum_i=0^m alpha_i = 1.
*/
static
void leastSquares(int mk, double *Fvecs, double *alpha)
{
    char trans = 'N';
    int rows = g_ESize;
    int cols = mk;
    int nrhs = 1;
    int lda = g_ESize;
    int ldb = g_ESize;
    int lwork = mk + g_ESize;
    int info = 0;
    
    
    // Put data into A
    for(int j = 0; j < mk; j++) {
    for(size_t i = 0; i < g_ESize; i++) {
        s_A[VECS(i,j)] = Fvecs[VECS(i,j)] - Fvecs[VECS(i,mk)];
    }}
    
    
    // Put data into b
    for(size_t i = 0; i < g_ESize; i++) {
        s_b[i] = -Fvecs[VECS(i,mk)];
    }
    
    
    // Do least squares
    dgels_(&trans, &rows, &cols, &nrhs, s_A, &lda, s_b, &ldb, s_work, &lwork, &info);
    if(info != 0) {
        printf("Least squares error from LAPACK: %d\n", info);
    }
    
    
    // Set alpha
    alpha[mk] = 1.0;
    for(int i = 0; i < mk; i++) {
        alpha[i] = s_b[i];
        alpha[mk] = alpha[mk] - alpha[i];
    }
}


/*
    Calculates 2-norm of vector.
*/
static
double calcNorm2(int n, double *v)
{
    double norm = 0.0;
    for(int i = 0; i < n; i++) {
        norm += v[i] * v[i];
    }
    
    return sqrt(norm);
}


/*
    Computes f where
    v dx f + E dv f + sigma f = q
    dx E = rho - rho0
*/
void eulerStepNonlinear(double sigma, double *E0, double *q, double *f)
{
    double *Evecs = s_Evecs;
    double *Fvecs = s_Fvecs;
    double *G = s_G;
    double *E = s_E;
    double *alpha = s_alpha;
    int k;
    double tol, error;
    int m = g_eulerRestart;
    
    
    // Setup Evecs
    for(size_t i = 0; i < g_ESize; i++) {
        Evecs[VECS(i,0)] = E0[i];
    }
    Gfunc(sigma, &Evecs[VECS(0,0)], q, f, &Evecs[VECS(0,1)], true);
    
    
    // Setup Fvecs
    for(size_t i = 0; i < g_ESize; i++) {
        Fvecs[VECS(i,0)] = Evecs[VECS(i,1)] - Evecs[VECS(i,0)];
    }
    
    
    // Setup tolerance
    tol = g_eulerTolA + g_eulerTolR * calcNorm2(g_ESize, &Fvecs[VECS(0,0)]);
    
    
    // Loop until nonlinear convergence
    k = 1;
    while(k < g_eulerMaxiter) {
        
        int mk = (k < m) ? k : m;
        int lastIndex;
        
        
        // Fill in next column of Fvecs
        Gfunc(sigma, &Evecs[VECS(0,mk)], q, f, G);
        for(size_t i = 0; i < g_ESize; i++) {
            Fvecs[VECS(i,mk)] = G[i] - Evecs[VECS(i,mk)];
        }
        
        
        // Check error
        error = calcNorm2(g_ESize, &Fvecs[VECS(0,mk)]);
        if(error < tol) {
            break;
        }
        
        
        // Solve least squares
        leastSquares(mk, Fvecs, alpha);
        
        
        // Find E_{k+1}
        for(size_t i = 0; i < g_ESize; i++) {
            E[i] = 0.0;
            for(int j = 0; j <= mk; j++) {
                E[i] += alpha[j] * (Fvecs[VECS(i,j)] + Evecs[VECS(i,j)]);
            }
        }
        
        
        // Shift Evecs, Fvecs
        if(mk == m) {
            shift(m, Evecs);
            shift(m, Fvecs);
            lastIndex = m;
        }
        else {
            lastIndex = mk + 1;
        }
        
        
        // Put last E_{k+1} into Evecs
        for(size_t i = 0; i < g_ESize; i++) {
            Evecs[VECS(i,lastIndex)] = E[i];
        }
        
        
        // Increment k
        k++;
    }
    
    
    if(g_printEuler)
        printf("   Nonlinear Euler Step: iters = %d\n", k);
    g_eulerIters.push_back(k);
}

