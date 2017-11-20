#include "global.h"
#include "input_deck_reader.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


/*
    Checks the input deck for an error.
    Kills the program if there is an error.
*/
static
void check(bool isOk, int line)
{
    if(!isOk) {
        printf("init.cpp: Input error at line %d\n", line);
        exit(0);
    }
}


/*
    Helper function for deciding if x is in a certain range.
*/
static
bool isIn2DSquare(double x)
{
    if((x > -0.7 && x < -0.4) || (x > 0.4 && x < 0.7))
        return true;
    return false;
}


/*
    2D Landau Initial Condition
*/
static
double landau2d(double x1, double x2, double v1, double v2)
{
    double k1 = 0.5;
    double k2 = 0.5;
    double alpha = 0.01;
    
    return (1.0 + alpha * (cos(k1*x1) + cos(k2*x2))) / (2.0*M_PI) 
           * exp(-(v1*v1 + v2*v2)/2.0);
}


/*
    1D Landau Initial Condition
*/
static
double landau1d(double x, double v)
{
    double k = 0.5;
    double alpha = 0.01;
    
    return (1.0 + alpha * cos(k*x)) / sqrt(2.0*M_PI) * exp(-v*v/2.0);
}


/*
    Two stream instability initial condition
*/
static
double twostream(double x, double v)
{
    return v*v / sqrt(8.0*M_PI) * (2.0 - cos(x/2.0)) * exp(-v*v/2.0);
}


/*
    Initialization routine
*/
void init(double *dt, double *tend, double **f_, double **rho_, double **E_)
{
    double *f, *rho, *E;
    InputDeckReader deck;
    char runType[100];
    
    
    // Read input deck
    deck.readInputDeck("input.deck");
    check(deck.getValue("dt", dt), __LINE__);
    check(deck.getValue("tend", tend), __LINE__);
    check(deck.getValue("nx", &g_nx), __LINE__);
    check(deck.getValue("nv", &g_nv), __LINE__);
    check(deck.getValue("use_new_sweep", &g_useNewSweep), __LINE__);
    check(deck.getValue("time_order", &g_timeOrder), __LINE__);
    check(deck.getValue("sweep_tol", &g_sweepTol), __LINE__);
    check(deck.getValue("sweep1_tol", &g_sweep1Tol), __LINE__);
    check(deck.getValue("sweep_maxiter", &g_sweepMaxiter), __LINE__);
    check(deck.getValue("sweep1_maxiter", &g_sweep1Maxiter), __LINE__);
    check(deck.getValue("efield_tol", &g_efieldTol), __LINE__);
    check(deck.getValue("euler_tola", &g_eulerTolA), __LINE__);
    check(deck.getValue("euler_tolr", &g_eulerTolR), __LINE__);
    check(deck.getValue("euler_maxiter", &g_eulerMaxiter), __LINE__);
    check(deck.getValue("euler_restart", &g_eulerRestart), __LINE__);
    check(deck.getValue("efield_maxiter", &g_efieldMaxiter), __LINE__);
    check(deck.getValue("run_type", runType), __LINE__);
    check(deck.getValue("use_nonlinear", &g_useNonlinear), __LINE__);
    check(deck.getValue("print_conserved", &g_printConserved), __LINE__);
    check(deck.getValue("print_sweep", &g_printSweep), __LINE__);
    check(deck.getValue("print_efield", &g_printEfield), __LINE__);
    check(deck.getValue("print_euler", &g_printEuler), __LINE__);
    check(deck.getValue("print_petsc", &g_printPETSC), __LINE__);
    check(deck.getValue("print_cfl", &g_printCFL), __LINE__);
    check(deck.getValue("upwind_flux", &g_upwindFlux), __LINE__);
    check(deck.getValue("precondition", &g_precondition), __LINE__);
    check(deck.getValue("estimate_ps_norm", &g_estimatePSNorm), __LINE__);
    
    
    // Set run type
    if(strcmp(runType, "efield_test") == 0) 
        g_runType = RUN_EFIELD_TEST;
    else if(strcmp(runType, "advection_squares") == 0) 
        g_runType = RUN_ADVECTION_SQUARES;
    else if(strcmp(runType, "advection_random") == 0) 
        g_runType = RUN_ADVECTION_RANDOM;
    else if(strcmp(runType, "two_stream") == 0) 
        g_runType = RUN_TWO_STREAM;
    else if(strcmp(runType, "landau_damping") == 0) 
        g_runType = RUN_LANDAU_DAMPING;
    else if(strcmp(runType, "convergence") == 0) 
        g_runType = RUN_CONVERGENCE;
    else if(strcmp(runType, "advection_squares_2d11") == 0) 
        g_runType = RUN_ADVECTION_SQUARES11;
    else if(strcmp(runType, "advection_squares_2d12") == 0) 
        g_runType = RUN_ADVECTION_SQUARES12;
    else if(strcmp(runType, "advection_squares_2d21") == 0) 
        g_runType = RUN_ADVECTION_SQUARES21;
    else if(strcmp(runType, "advection_squares_2d22") == 0) 
        g_runType = RUN_ADVECTION_SQUARES22;
    else if(strcmp(runType, "landau_2d") == 0) 
        g_runType = RUN_LANDAU_2D;
    else if(strcmp(runType, "efield_test_2d") == 0) 
        g_runType = RUN_EFIELD_TEST_2D;
    else {
        printf("Run Type unrecognized\n");
        exit(0);
    }
    
    
    // Set dimension
    if(g_runType < NUM_RUNS_1D)
        g_dimension = 1;
    else
        g_dimension = 2;
    
    
    // Set basis sizes.
    if(g_dimension == 1) {
        g_nBasis = 3;
        g_nBasisBdry = 2;
        g_nBasisRho = 2;
        g_rhoSize = g_nx * 2;
        g_fSize = g_nx * g_nv * 3;
        g_ESize = g_nx;
    }
    else {
        g_nBasis = 5;
        g_nBasisBdry = 4;
        g_nBasisRho = 3;
        g_rhoSize = g_nx * g_nx * 3;
        g_fSize = g_nx * g_nx * g_nv * g_nv * 5;
        g_ESize = g_nx * g_nx * 2;
    }
    
    
    // Check to make sure 2D problem isn't too big.
    // Makes sure your system doesn't get completely hosed.
    if(g_fSize * 8 > 250000000) {
        printf("init.cpp: fSize may be too large  %ld MB.\n", 
               g_fSize * 8 / 1000000);
        exit(0);
    }
    
    
    // Check for efield test
    if(g_runType == RUN_EFIELD_TEST) {
        printf("\n--- 1D Efield Test ---\n");
        g_isPeriodic = true;
        g_nx = 50;  efield_test();
        g_nx = 100; efield_test();
        g_nx = 200; efield_test();
        g_nx = 400; efield_test();
        
        exit(0);
    }
    if(g_runType == RUN_EFIELD_TEST_2D) {
        printf("\n--- 2D Efield Test ---\n");
        g_isPeriodic = true;
        g_nx = 20;  efield_test();
        g_nx = 40;  efield_test();
        g_nx = 80;  efield_test();
        g_nx = 160; efield_test();
        
        exit(0);
    }
    
    
    // Set domain parameters
    switch(g_runType) {
        case RUN_ADVECTION_SQUARES:
        case RUN_ADVECTION_RANDOM:
        case RUN_ADVECTION_SQUARES11:
        case RUN_ADVECTION_SQUARES12:
        case RUN_ADVECTION_SQUARES21:
        case RUN_ADVECTION_SQUARES22:
            g_ax = -1.2;
            g_av = -1.2;
            g_bx = 1.2;
            g_bv = 1.2;
            g_isPeriodic = true;
            g_calcEfield = false;
            break;
        
        case RUN_TWO_STREAM:
            g_ax = -2.0 * M_PI;
            g_av = -2.0 * M_PI;
            g_bx = 2.0 * M_PI;
            g_bv = 2.0 * M_PI;
            g_isPeriodic = true;
            g_calcEfield = true;
            break;
        
        case RUN_LANDAU_DAMPING:
        case RUN_LANDAU_2D:
            g_ax = -2.0 * M_PI;
            g_av = -2.0 * M_PI;
            g_bx = 2.0 * M_PI;
            g_bv = 2.0 * M_PI;
            g_isPeriodic = true;
            g_calcEfield = true;
            break;
        
        case RUN_CONVERGENCE:
            g_ax = -M_PI;
            g_av = -M_PI;
            g_bx = M_PI;
            g_bv = M_PI;
            g_isPeriodic = true;
            g_calcEfield = true;
            break;
        
        default:
            break;
    }
    
    
    // Set dx, dv
    g_dx = (g_bx - g_ax) / g_nx;
    g_dv = (g_bv - g_av) / g_nv;
    
    
    // Allocate memory
    *f_   = f   = new double[g_fSize];
    *rho_ = rho = new double[g_rhoSize];
    *E_   = E   = new double[g_ESize];
    g_x = new double[g_nx];
    g_v = new double[g_nv];
    
    
    // Set g_x, g_v
    for(int i = 0; i < g_nx; i++) {
        g_x[i] = g_ax + g_dx / 2.0 + g_dx * i;
    }
    for(int j = 0; j < g_nv; j++) {
        g_v[j] = g_av + g_dv / 2.0 + g_dv * j;
    }
    
    
    // Initial condition for f
    switch(g_runType) {
        case RUN_ADVECTION_SQUARES:
            for(int i = 0; i < g_nx; i++) {
            for(int j = 0; j < g_nv; j++) {
                if(isIn2DSquare(g_x[i]) && isIn2DSquare(g_v[j])) {
                    f[IJK(i,j,0)] = 1.0;
                }
                else {
                    f[IJK(i,j,0)] = 0.0;
                }
                
                f[IJK(i,j,1)] = 0.0;
                f[IJK(i,j,2)] = 0.0;
            }}
            break;
            
        case RUN_ADVECTION_RANDOM:
            for(int i = 0; i < g_nx; i++) {
            for(int j = 0; j < g_nv; j++) {
                f[IJK(i,j,0)] = rand();
                f[IJK(i,j,1)] = 0.0;
                f[IJK(i,j,2)] = 0.0;
            }}
            break;
        
        case RUN_TWO_STREAM:
            for(int i = 0; i < g_nx; i++) {
            for(int j = 0; j < g_nv; j++) {
                double x1, x2, v1, v2;
                double f11, f12, f21, f22;
                
                x1 = g_x[i] - g_dx / 2.0;
                x2 = g_x[i] + g_dx / 2.0;
                v1 = g_v[j] - g_dv / 2.0;
                v2 = g_v[j] + g_dv / 2.0;
                
                f11 = twostream(x1,v1);
                f12 = twostream(x1,v2);
                f21 = twostream(x2,v1);
                f22 = twostream(x2,v2);
                
                f[IJK(i,j,0)] = 0.25 * (f22 + f21 + f12 + f11);
                f[IJK(i,j,1)] = 0.25 * (f22 + f21 - f12 - f11);
                f[IJK(i,j,2)] = 0.25 * (f22 - f21 + f12 - f11);
            }}
            break;
        
        case RUN_LANDAU_DAMPING:
            for(int i = 0; i < g_nx; i++) {
            for(int j = 0; j < g_nv; j++) {
                double x1, x2, v1, v2;
                double f11, f12, f21, f22;
                
                x1 = g_x[i] - g_dx / 2.0;
                x2 = g_x[i] + g_dx / 2.0;
                v1 = g_v[j] - g_dv / 2.0;
                v2 = g_v[j] + g_dv / 2.0;
                
                f11 = landau1d(x1,v1);
                f12 = landau1d(x1,v2);
                f21 = landau1d(x2,v1);
                f22 = landau1d(x2,v2);
                
                f[IJK(i,j,0)] = 0.25 * (f22 + f21 + f12 + f11);
                f[IJK(i,j,1)] = 0.25 * (f22 + f21 - f12 - f11);
                f[IJK(i,j,2)] = 0.25 * (f22 - f21 + f12 - f11);
            }}
            break;
        
        case RUN_CONVERGENCE:
            for(int i = 0; i < g_nx; i++) {
            for(int j = 0; j < g_nv; j++) {
                double x1, x2, v1, v2;
                double f11, f12, f21, f22;
                
                x1 = g_x[i] - g_dx / 2.0;
                x2 = g_x[i] + g_dx / 2.0;
                v1 = g_v[j] - g_dv / 2.0;
                v2 = g_v[j] + g_dv / 2.0;
                
                f11 = conv_solution(x1, v1, 0.0);
                f12 = conv_solution(x1, v2, 0.0);
                f21 = conv_solution(x2, v1, 0.0);
                f22 = conv_solution(x2, v2, 0.0);
                
                f[IJK(i,j,0)] = 0.25 * (f22 + f21 + f12 + f11);
                f[IJK(i,j,1)] = 0.25 * (f22 + f21 - f12 - f11);
                f[IJK(i,j,2)] = 0.25 * (f22 - f21 + f12 - f11);
            }}
            break;
        
        case RUN_ADVECTION_SQUARES11:
        case RUN_ADVECTION_SQUARES21:
        case RUN_ADVECTION_SQUARES12:
        case RUN_ADVECTION_SQUARES22:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
            for(int j1 = 0; j1 < g_nv; j1++) {
            for(int j2 = 0; j2 < g_nv; j2++) {
                if(isIn2DSquare(g_x[i1]) && isIn2DSquare(g_x[i2]) && 
                   isIn2DSquare(g_v[j1]) && isIn2DSquare(g_v[j2]))
                {
                    f[IIJJK(i1,i2,j1,j2,0)] = 1.0;
                }
                else {
                    f[IIJJK(i1,i2,j1,j2,0)] = 0.0;
                }
                
                f[IIJJK(i1,i2,j1,j2,1)] = 0.0;
                f[IIJJK(i1,i2,j1,j2,2)] = 0.0;
                f[IIJJK(i1,i2,j1,j2,3)] = 0.0;
                f[IIJJK(i1,i2,j1,j2,4)] = 0.0;
            }}}}
            break;
        
        case RUN_LANDAU_2D:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
            for(int j1 = 0; j1 < g_nv; j1++) {
            for(int j2 = 0; j2 < g_nv; j2++) {
                double x1m, x1p, x2m, x2p;
                double v1m, v1p, v2m, v2p;
                
                x1m = g_x[i1] - g_dx / 2.0;
                x1p = g_x[i1] + g_dx / 2.0;
                x2m = g_x[i2] - g_dx / 2.0;
                x2p = g_x[i2] + g_dx / 2.0;
                v1m = g_v[i1] - g_dv / 2.0;
                v1p = g_v[i1] + g_dv / 2.0;
                v2m = g_v[i2] - g_dv / 2.0;
                v2p = g_v[i2] + g_dv / 2.0;
                
                f[IIJJK(i1,i2,j1,j2,0)] = landau2d(g_x[i1], g_x[i2], g_v[j1], g_v[j2]);
                f[IIJJK(i1,i2,j1,j2,1)] = 0.5 * (landau2d(x1p, g_x[i2], g_v[j1], g_v[j2])
                                        - landau2d(x1m, g_x[i2], g_v[j1], g_v[j2]));
                f[IIJJK(i1,i2,j1,j2,2)] = 0.5 * (landau2d(g_x[i1], x2p, g_v[j1], g_v[j2])
                                        - landau2d(g_x[i1], x2m, g_v[j1], g_v[j2]));
                f[IIJJK(i1,i2,j1,j2,3)] = 0.5 * (landau2d(g_x[i1], g_x[i2], v1p, g_v[j2])
                                        - landau2d(g_x[i1], g_x[i2], v1m, g_v[j2]));
                f[IIJJK(i1,i2,j1,j2,4)] = 0.5 * (landau2d(g_x[i1], g_x[i2], g_v[j1], v2p)
                                        - landau2d(g_x[i1], g_x[i2], g_v[j1], v2m));
            }}}}
            break;
        
        default:
            break;
    }
    
    
    // Initial condition for E for advection of squares
    switch(g_runType) {
        case RUN_ADVECTION_SQUARES:
        case RUN_ADVECTION_RANDOM:
            for(int i = 0; i < g_nx; i++) {
                E[i] = -g_x[i];
            }
            break;
        
        case RUN_ADVECTION_SQUARES11:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
                E[NII(0,i1,i2)] = -g_x[i1];
                E[NII(1,i1,i2)] = 0.0;
            }}
            break;
        case RUN_ADVECTION_SQUARES12:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
                E[NII(0,i1,i2)] = 0.0;
                E[NII(1,i1,i2)] = -g_x[i1];
            }}
            break;
        case RUN_ADVECTION_SQUARES21:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
                E[NII(0,i1,i2)] = -g_x[i2];
                E[NII(1,i1,i2)] = 0.0;
            }}
            break;
        case RUN_ADVECTION_SQUARES22:
            for(int i1 = 0; i1 < g_nx; i1++) {
            for(int i2 = 0; i2 < g_nx; i2++) {
                E[NII(0,i1,i2)] = 0.0;
                E[NII(1,i1,i2)] = -g_x[i2];
            }}
            break;
        
        default:
            break;
    }
}


