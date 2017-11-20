#pragma once

#include <vector>
#include <string.h>


// 1D macros
#define IK(i,k) ((i) * g_nBasisBdry + (k))
#define JK(j,k) ((j) * g_nBasisBdry + (k))
#define IJK(i,j,k) (((i) * g_nv + (j)) * g_nBasis + (k))

// 2D macros
#define II(i1,i2) ( (i1) * g_nx + (i2) )
#define JJ(j1,j2) ( (j1) * g_nv + (j2) )
#define IIJJK(i1,i2,j1,j2,k) ( (II(i1,i2) * g_nv * g_nv + JJ(j1,j2)) * g_nBasis + (k) )
#define IIK(i1,i2,k) ( II(i1,i2) * g_nBasisRho + (k) )
#define NII(n,i1,i2) ( (n) * g_nx * g_nx + II(i1,i2) )
#define IJJK(i,j1,j2,k) ( ((i) * g_nv * g_nv + JJ(j1,j2)) * g_nBasisBdry + (k) )
#define IIJK(i1,i2,j,k) ( (II(i1,i2) * g_nv + (j)) * g_nBasisBdry + (k) )


// Trick to define extern variables in extern.cpp
#ifdef NO_EXTERN
#define EXTERN
#else
#define EXTERN extern
#endif


// Different run types
enum RUN_TYPE
{
    RUN_EFIELD_TEST, RUN_ADVECTION_SQUARES, RUN_ADVECTION_RANDOM, 
    RUN_TWO_STREAM, RUN_LANDAU_DAMPING, RUN_CONVERGENCE,
    NUM_RUNS_1D,
    RUN_ADVECTION_SQUARES11, RUN_ADVECTION_SQUARES12, RUN_ADVECTION_SQUARES21, 
    RUN_ADVECTION_SQUARES22, RUN_LANDAU_2D, RUN_EFIELD_TEST_2D
};


// Data to be stored at each time step
struct TimeStepData
{
    double time;
    double l2efield1;
    double l2efield2;
    double mass;
    double momentum1;
    double momentum2;
    double energy;
    double l2norm;
    double minECFL;
    double maxECFL;
    double meanECFL;
    double xCFL;
};


// Global variables
EXTERN int g_nx, g_nv, g_nBasis, g_nBasisBdry, g_nBasisRho;
EXTERN double g_ax, g_bx, g_av, g_bv;
EXTERN double g_dx, g_dv;
EXTERN double *g_x, *g_v;
EXTERN bool g_isPeriodic;
EXTERN bool g_useNewSweep;
EXTERN int g_timeOrder;
EXTERN double g_sweepTol, g_sweep1Tol;
EXTERN int g_sweepMaxiter, g_sweep1Maxiter;
EXTERN double g_efieldTol;
EXTERN double g_eulerTolA, g_eulerTolR;
EXTERN int g_eulerMaxiter;
EXTERN int g_eulerRestart;
EXTERN int g_efieldMaxiter;
EXTERN RUN_TYPE g_runType;
EXTERN bool g_useNonlinear;
EXTERN std::vector<TimeStepData> g_timeStepData;
EXTERN std::vector<int> g_efieldIters;
EXTERN std::vector<int> g_sweepIters;
EXTERN std::vector<int> g_eulerIters;
EXTERN bool g_printConserved;
EXTERN bool g_printSweep;
EXTERN bool g_printEfield;
EXTERN bool g_printEuler;
EXTERN bool g_printPETSC;
EXTERN bool g_printCFL;
EXTERN bool g_upwindFlux;
EXTERN bool g_precondition;
EXTERN size_t g_rhoSize;
EXTERN size_t g_fSize;
EXTERN size_t g_ESize;
EXTERN int g_dimension;
EXTERN bool g_estimatePSNorm;
EXTERN bool g_calcEfield;


//// Global Functions ////

// From sweep1d.cpp
void sweep1d_init();
void sweep1d_end();
void sweep1d(double sigma, double *E, double *q, double *f);


// From sweep1d_fast.cpp
void sweep1dFast_init();
void sweep1dFast_end();
void sweep1dFast(double sigma, double *E, double *q, double *f);
void estimateNormPS1d();


// From sweep2d.cpp
void sweep2d_init();
void sweep2d_end();
void sweep2d(double sigma, double *E, double *q, double *f);


// From sweep2d_fast.cpp
void sweep2dFast_init();
void sweep2dFast_end();
void sweep2dFast(double sigma, double *E, double *q, double *f);
void estimateNormPS2d();


// From efield.cpp
void efield_init();
void efield_end();
void efield(double *rho, double *E);
void efield_test();


// From output.cpp
void output(const char *filename, double *f);
void outputTimeStepData(const char *filename);
void outputEfield(const char *filename, const int n, const double *phi, 
                  const double *E1, const double *E2);


// From init.cpp
void init(double *dt, double *tend, double **f_, double **rho_, double **E_);


// From timestep.cpp
void timestep_init();
void timestep_end();
void timestep(double t, double dt, double *E, double *f);


// from timestep_data.cpp
void calcTimeStepQuantities1D(double dt, double t, double *f, double *E);
void calcTimeStepQuantities2D(double dt, double t, double *f, double *E);


// From euler_step.cpp
void eulerStep_init();
void eulerStep_end();
void eulerStep(double t, double dt, double sigma, double *E0, double *q, double *f);


// From euler_step_nonlinear.cpp
void eulerStepNonlinear_init();
void eulerStepNonlinear_end();
void eulerStepNonlinear(double sigma, double *E0, double *q, double *f);


// From utils.cpp
double conv_solution(double x, double v, double t);
double conv_efield(double x, double t);
double conv_source(double x, double v, double t);
void conv_error(double *f, double *E, double t);
void calcRho(double *f, double *rho);
void solveNxN(int n, double *A_, double *b);



