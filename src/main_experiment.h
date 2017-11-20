#include "global.h"
#include <petscksp.h>
#include <stdio.h>


/*
    Program begin.
*/
int main(int argc, char **argv)
{
    // Variables
    double tend;
    double dt;
    double *f, *rho, *E;
    double t;
    bool lastTimeStep = false;
    PetscErrorCode err;
    
    
    // Initialize Petsc
    err = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(err);
    
    
    // Initialize solver
    init(&dt, &tend, &f, &rho, &E);
    
    
    // Initialize subsystems
    if(g_dimension == 1) {
        if(!g_useNewSweep)
            sweep1dSplit_init();
            //sweep1d_init();
        if(g_useNewSweep || g_precondition)
            sweep1dFast_init();
    }
    else {
        if(!g_useNewSweep)
            sweep2d_init();
        if(g_useNewSweep || g_precondition)
            sweep2dFast_init();
    }
    if(g_calcEfield)
        efield_init();
    timestep_init();
    if(g_useNonlinear)
        eulerStepNonlinear_init();
    else
        eulerStep_init();
    
    
    // Output initial condition
    output("begin.dat", f);
    
    
    // Update in time
    t = 0.0;
    while(true) {
        
        // Update dt at last time step
        if(t + dt >= tend * (1.0 - 1e-10)) {
            dt = tend - t;
            lastTimeStep = true;
        }
        
        
        // Print time and conserved quantities
        printf("t = %.7f, dt = %.7f\n", t, dt);
        
        
        // Get e-field
        if(g_calcEfield) {
            calcRho(f, rho);
            efield(rho, E);
        }
        
        
        // Save conserved quantities and efield
        if(g_dimension == 1)
            calcTimeStepQuantities1D(dt, t, f, E);
        else
            calcTimeStepQuantities2D(dt, t, f, E);
        
        
        // Time step
        timestep(t, dt, E, f);
        
        
        // Update t
        t = t + dt;
        if(lastTimeStep)
            break;
    }
    
    
    // Print time
    printf("t = %.7f\n", t);
    
    
    // Get e-field
    if(g_calcEfield) {
        calcRho(f, rho);
        efield(rho, E);
    }
    
    
    // Save conserved quantities and efield
    if(g_dimension == 1)
        calcTimeStepQuantities1D(dt, t, f, E);
    else
        calcTimeStepQuantities2D(dt, t, f, E);
    
    
    // Output end condition
    output("end.dat", f);
    outputTimeStepData("timeStepData.dat");
    
    
    // Output error if convergence
    if(g_runType == RUN_CONVERGENCE) {
        conv_error(f, E, t);
    }
    
    
    // Cleanup
    delete[] f;
    delete[] rho;
    delete[] E;
    
    if(g_dimension == 1) {
        if(!g_useNewSweep)
            sweep1d_end();
        if(g_useNewSweep || g_precondition)
            sweep1dFast_end();
    }
    else {
        if(!g_useNewSweep)
            sweep2d_end();
        if(g_useNewSweep || g_precondition)
            sweep2dFast_end();
    }
    if(g_calcEfield)
        efield_end();
    timestep_end();
    if(g_useNonlinear)
        eulerStepNonlinear_end();
    else
        eulerStep_end();
    
    PetscFinalize();
    return 0;
}



