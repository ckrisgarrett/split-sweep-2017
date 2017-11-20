/*
    Experimental File that solves the poisson equation by replacing
    the last row of the matrix with all ones and the last entry of b with 0.0
*/


#include "global.h"
#include <petscksp.h>


// Indexing for 2D phi (has 1 less entry in each dimension than E)
#define II1(i1,i2) ( (i1) * (g_nx) + (i2) - 1)


static KSP s_ksp;
static Mat s_A;
static Vec s_b, s_f;
static bool s_isEfieldTest = false;


/*
    Setup Diffusion Matrix
*/
static
void setupMatrix(Mat A)
{
    PetscErrorCode err;
    
    
    // Set entire matrix to zero.
    err = MatZeroEntries(A);CHKERRV(err);
    
    
    // 1D matrix
    if(g_dimension == 1) {
        double diag = -2.0 / (g_dx * g_dx);
        double offdiag = 1.0 / (g_dx * g_dx);
        
        for(int i = 1; i < g_nx-1; i++) {
            err = MatSetValue(A, i, i, diag, INSERT_VALUES);CHKERRV(err);
            err = MatSetValue(A, i, i-1, offdiag, INSERT_VALUES);CHKERRV(err);
            err = MatSetValue(A, i, i+1, offdiag, INSERT_VALUES);CHKERRV(err);
        }
        
        // Row 0
        err = MatSetValue(A, 0, 0, diag, INSERT_VALUES);CHKERRV(err);
        err = MatSetValue(A, 0, g_nx-1, offdiag, INSERT_VALUES);CHKERRV(err);
        err = MatSetValue(A, 0, 1, offdiag, INSERT_VALUES);CHKERRV(err);
        
        // Row g_nx-1
        for(int j = 0; j < g_nx; j++) {
            err = MatSetValue(A, g_nx-1, j, offdiag, INSERT_VALUES);CHKERRV(err);
        }
    }
    // 2D matrix
    else {
        double diag = -8.0 / 3.0 / (g_dx * g_dx);
        double offdiag = 1.0 / 3.0 / (g_dx * g_dx);
        
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            int i1m = (i1 == 0) ? g_nx-1 : i1-1;
            int i1p = (i1 == g_nx-1) ? 0 : i1+1;
            int i2m = (i2 == 0) ? g_nx-1 : i2-1;
            int i2p = (i2 == g_nx-1) ? 0 : i2+1;
            
            // Center
            err = MatSetValue(A, II(i1,i2), II(i1,i2), diag, INSERT_VALUES);CHKERRV(err);
            
            // Sides
            if(II(i1m,i2) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1m,i2), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1p,i2) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1p,i2), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1,i2m) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1,i2m), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1,i2p) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1,i2p), offdiag, INSERT_VALUES);CHKERRV(err);
            
            // Diagonals
            if(II(i1m,i2m) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1m,i2m), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1m,i2p) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1m,i2p), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1p,i2m) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1p,i2m), offdiag, INSERT_VALUES);CHKERRV(err);
            if(II(i1p,i2p) >= 0)
                err = MatSetValue(A, II(i1,i2), II(i1p,i2p), offdiag, INSERT_VALUES);CHKERRV(err);
        }}
        
        // Last row
        for(int j = 0; j < g_nx * g_nx; j++) {
            err = MatSetValue(A, g_nx*g_nx-1, j, offdiag, INSERT_VALUES);CHKERRV(err);
        }
    }
    
    
    // Assemble matrix
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}


/*
    Init
*/
void efield_init()
{
    int n, stencilSize;
    PetscErrorCode err;
    
    
    // Size depending on dimension
    if(g_dimension == 1) {
        n = g_nx;
        //stencilSize = 3;
        stencilSize = n;
    }
    else {
        n = g_nx * g_nx;
        //stencilSize = 9;
        stencilSize = n;
    }
    
    
    // Petsc setup
    err = MatCreateSeqAIJ(PETSC_COMM_WORLD, n, n, stencilSize, NULL, &s_A);CHKERRV(err);
    err = MatSetOption(s_A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRV(err);
    setupMatrix(s_A);
    
    err = VecCreate(PETSC_COMM_WORLD, &s_b);CHKERRV(err);
    err = VecSetType(s_b, VECSEQ);CHKERRV(err);
    err = VecSetSizes(s_b, PETSC_DECIDE, n);CHKERRV(err);
    err = VecDuplicate(s_b, &s_f);CHKERRV(err);
    
    err = KSPCreate(PETSC_COMM_WORLD, &s_ksp);CHKERRV(err);
    err = KSPSetOperators(s_ksp,s_A,s_A);CHKERRV(err);
    
    err = KSPSetTolerances(s_ksp, g_efieldTol, g_efieldTol*g_efieldTol, PETSC_DEFAULT, 
                           g_efieldMaxiter);CHKERRV(err);
    err = KSPSetInitialGuessNonzero(s_ksp, PETSC_TRUE);CHKERRV(err);
    //err = KSPSetType(s_ksp, KSPCG);CHKERRV(err);
    err = KSPSetType(s_ksp, KSPGMRES);CHKERRV(err);
    
    //PC prec;
    //err = KSPGetPC(s_ksp, &prec);CHKERRV(err);
    //err = PCSetType(prec, PCNONE);CHKERRV(err);
    //err = PCSetType(prec, PCILU);CHKERRV(err);
    //err = PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE);CHKERRV(err);
}


/* 
    Free data.
*/
void efield_end()
{
    PetscErrorCode err;
    
    err = VecDestroy(&s_b);CHKERRV(err);
    err = VecDestroy(&s_f);CHKERRV(err);
    err = MatDestroy(&s_A);CHKERRV(err);
    err = KSPDestroy(&s_ksp);CHKERRV(err);
}


/*
    Calculates residual of Krylov solve.
*/
static 
void residual(Mat A, Vec x, Vec y, double *res)
{
    Vec y1;
    PetscReal normy, normx;
    PetscErrorCode err;
    
    err = VecDuplicate(y, &y1);CHKERRV(err);
    err = MatMult(A, x, y1);CHKERRV(err);
    err = VecAXPBY(y1, 1.0, -1.0, y);CHKERRV(err);
    err = VecNorm(y1, NORM_1, &normy);CHKERRV(err);
    err = VecNorm(x,  NORM_1, &normx);CHKERRV(err);
    
    err = VecDestroy(&y1);CHKERRV(err);
    
    *res = normy / normx;
}


/*
    Performs efield solve.
*/
void efield(double *rho, double *E)
{
    int iter;
    PetscErrorCode err;
    double *temp;
    double rho0;
    KSPConvergedReason reason;
    double res;
    
    
    // Calculate rho0
    if(g_dimension == 1) {
        rho0 = 0.0;
        for(int i = 0; i < g_nx; i++) {
            rho0 += g_dx * rho[IK(i,0)];
        }
        rho0 = rho0 / (g_bx - g_ax);
    }
    else {
        rho0 = 0.0;
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            rho0 += g_dx * g_dx * rho[IIK(i1,i2,0)];
        }}
        rho0 = rho0 / (g_bx - g_ax) / (g_bx - g_ax);
    }
    
    
    // Calculate s_b = rho - rho0
    err = VecGetArray(s_b, &temp);CHKERRV(err);
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx-1; i++) {
            int im = (i == 0) ? g_nx-1 : i-1;
            temp[i] = 0.5 * (rho[IK(im,0)] + rho[IK(im,1)] / 3.0)
                    + 0.5 * (rho[IK(i,0)] - rho[IK(i,1)] / 3.0) - rho0;
        }
        temp[g_nx-1] = 0.0;
    }
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            int i1m = (i1 == 0) ? g_nx-1 : i1-1;
            int i2m = (i2 == 0) ? g_nx-1 : i2-1;
            temp[II(i1,i2)] = -rho0
                + 0.25 * (rho[IIK(i1m,i2m,0)] + rho[IIK(i1m,i2m,1)] / 3.0 + rho[IIK(i1m,i2m,2)] / 3.0)
                + 0.25 * (rho[IIK(i1, i2m,0)] - rho[IIK(i1, i2m,1)] / 3.0 + rho[IIK(i1, i2m,2)] / 3.0)
                + 0.25 * (rho[IIK(i1m,i2, 0)] + rho[IIK(i1m,i2, 1)] / 3.0 - rho[IIK(i1m,i2, 2)] / 3.0)
                + 0.25 * (rho[IIK(i1, i2, 0)] - rho[IIK(i1, i2, 1)] / 3.0 - rho[IIK(i1, i2, 2)] / 3.0);
        }}
        temp[g_nx*g_nx-1] = 0.0;
    }
    err = VecRestoreArray(s_b, &temp);CHKERRV(err);
    
    
    // Calculate phi
    err = KSPSolve(s_ksp, s_b, s_f);CHKERRV(err);
    if(g_printPETSC)
        err = KSPView(s_ksp, PETSC_VIEWER_STDOUT_WORLD);CHKERRV(err);
    err = KSPGetIterationNumber(s_ksp, &iter);CHKERRV(err);
    err = KSPGetConvergedReason(s_ksp, &reason);CHKERRV(err);
    residual(s_A, s_f, s_b, &res);
    
    
    // Print CG stats
    if(g_printEfield) {
        printf("   efield: CG stats... iter: %d   residual: %e   reason: %s\n", 
               iter, res, KSPConvergedReasons[reason]);
    }
    g_efieldIters.push_back(iter);
    
    
    // Calculate E
    err = VecGetArray(s_f, &temp);CHKERRV(err);
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx; i++) {
            double phi1 = temp[i];
            double phi2 = (i == g_nx-1) ? temp[0] : temp[i+1];
            E[i] = (phi2 - phi1) / g_dx;
        }
        
        if(s_isEfieldTest) {
            outputEfield("efield.dat", g_nx, temp, E, NULL);
        }
    }
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            double phi11, phi12, phi21, phi22;
            int i1p, i2p;
            
            i1p = (i1 == g_nx-1) ? 0 : i1+1;
            i2p = (i2 == g_nx-1) ? 0 : i2+1;
            
            phi11 = temp[II(i1,i2)];
            phi12 = temp[II(i1,i2p)];
            phi21 = temp[II(i1p,i2)];
            phi22 = temp[II(i1p,i2p)];
            
            E[NII(0,i1,i2)] = 0.5 * (phi21 + phi22 - phi11 - phi12) / g_dx;
            E[NII(1,i1,i2)] = 0.5 * (phi22 + phi12 - phi11 - phi21) / g_dx;
        }}
        
        if(s_isEfieldTest) {
            outputEfield("efield.dat", g_nx*g_nx, temp, &E[NII(0,0,0)], &E[NII(1,0,0)]);
        }
    }
    err = VecRestoreArray(s_f, &temp);CHKERRV(err);
}


/*
    E1 Error: Calculates efield test 2D E1 error.
*/
static
double E1Error(double a1, double a2, double b1, double b2, double E1)
{
    return 0.25 * ((b1-a1) / 2.0 - (sin(2*b1)-sin(2*a1)) / 4.0)
                * 0.5 * ((b2-a2) + sin(b2)*cos(b2) - sin(a2)*cos(a2)) 
                + E1 / 2.0 * (cos(b1)-cos(a1)) * (sin(b2)-sin(a2)) 
                + E1 * E1 * (b1-a1) * (b2-a2);
}


/*
    Test efield solver.
        1D: rho = cos x
            E   = sin x
        2D: rho = cos(x1)*cos(x2)
            E   = (1/2 sin(x1)*cos(x2), 1/2 cos(x1)*sin(x2))
*/
void efield_test()
{
    double *rho, *EGuess, *EActual;
    double maxdiff = 0.0;
    double l2diff = 0.0;
    
    
    // Set ax, bx, dx
    g_ax = -M_PI;
    g_bx = M_PI;
    g_dx = 2 * M_PI / g_nx;
    g_x = new double[g_nx];
    for(int i = 0; i < g_nx; i++) {
        g_x[i] = g_ax + g_dx / 2.0 + g_dx * i;
    }
    
    
    // Set variable sizes
    if(g_dimension == 1) {
        g_rhoSize = g_nx * 2;
        g_ESize = g_nx;
    }
    else {
        g_rhoSize = g_nx * g_nx * 3;
        g_ESize = g_nx * g_nx * 2;
    }
    
    
    // Initialize
    efield_init();
    s_isEfieldTest = true;
    
    
    // Allocate memory
    rho = new double[g_rhoSize];
    EGuess = new double[g_ESize];
    EActual = new double[g_ESize];
    
    
    // Set initial condition and solution
    if(g_dimension == 1) {
        for(int i = 0; i < g_nx; i++) {
            rho[IK(i,0)] = cos(g_x[i]);
            rho[IK(i,1)] = 0.5 * (cos(g_x[i]+g_dx/2.0) - cos(g_x[i]-g_dx/2.0));
            
            EGuess[i] = 0.0;
            EActual[i] = sin(g_x[i]);
        }
    }
    else {
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
            rho[IIK(i1,i2,0)] = cos(g_x[i1]) * cos(g_x[i2]);
            rho[IIK(i1,i2,1)] = 0.5 * cos(g_x[i2]) 
                * (sin(g_x[i1]+g_dx/2) - sin(g_x[i1]-g_dx/2));
            rho[IIK(i1,i2,2)] = 0.5 * cos(g_x[i1]) 
                * (sin(g_x[i2]+g_dx/2) - sin(g_x[i2]-g_dx/2));
            
            EGuess[NII(0,i1,i2)] = EGuess[NII(1,i1,i2)] = 0.0;
            EActual[NII(0,i1,i2)] = 0.5 * sin(g_x[i1]) * cos(g_x[i2]);
            EActual[NII(1,i1,i2)] = 0.5 * cos(g_x[i1]) * sin(g_x[i2]);
        }}
    }
    
    
    // Solve for e-field
    efield(rho, EGuess);
    
    
    // Check accuracy
    // Max Diff
    maxdiff = 0.0;
    for(size_t i = 0; i < g_ESize; i++) {
        double diff = fabs(EActual[i] - EGuess[i]);
        if(diff > maxdiff)
            maxdiff = diff;
    }
    
    
    // L2 Diff
    if(g_dimension == 2) {
        l2diff = 0.0;
        for(int i1 = 0; i1 < g_nx; i1++) {
        for(int i2 = 0; i2 < g_nx; i2++) {
        
            double a1 = g_x[i1] - g_dx / 2.0;
            double a2 = g_x[i2] - g_dx / 2.0;
            double b1 = g_x[i1] + g_dx / 2.0;
            double b2 = g_x[i2] + g_dx / 2.0;
            double E1 = EGuess[NII(0,i1,i2)];
            //double E2 = EGuess[NII(1,i1,i2)];
        
            double diff = E1Error(a1, a2, b1, b2, E1);
            l2diff += diff * diff;
        }}
        
        l2diff = sqrt(l2diff);
    }
    
    
    if(g_dimension == 1)
        printf("MAX DIFF: %e\n", maxdiff);
    if(g_dimension == 2)
        printf("MAX DIFF: %e   E1 L2 DIFF: %e\n", maxdiff, l2diff);
    
    
    // Cleanup
    delete[] rho;
    delete[] EActual;
    delete[] EGuess;
    delete[] g_x;
    efield_end();
}







