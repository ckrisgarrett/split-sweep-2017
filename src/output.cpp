#include "global.h"
#include <stdio.h>


/*
    Output f
*/
void output(const char *filename, double *f)
{
    FILE *file;
    
    // Open file
    file = fopen(filename, "w");
    
    // Output data
    // NX, NV, NB
    fprintf(file, "%d\n", g_nx);
    fprintf(file, "%d\n", g_nv);
    fprintf(file, "%d\n", g_nBasis);
    
    // AX, BX, AV, BV
    fprintf(file, "%e\n", g_ax);
    fprintf(file, "%e\n", g_bx);
    fprintf(file, "%e\n", g_av);
    fprintf(file, "%e\n", g_bv);
    
    // f
    for(size_t index = 0; index < g_fSize; index++) {
        fprintf(file, "%e\n", f[index]);
    }
    
    // Close file
    fclose(file);
}


/*
    Output data collected at each time step.
*/
void outputTimeStepData(const char *filename)
{
    FILE *file;
    
    // Open file
    file = fopen(filename, "w");
    
    // Output num time steps, num efield iters, num sweep iters, num euler iters
    fprintf(file, "%ld\n", g_timeStepData.size());
    fprintf(file, "%ld\n", g_efieldIters.size());
    fprintf(file, "%ld\n", g_sweepIters.size());
    fprintf(file, "%ld\n", g_eulerIters.size());
    
    // Write time step data
    for(unsigned int i = 0; i < g_timeStepData.size(); i++) {
        TimeStepData tsd = g_timeStepData[i];
        
        fprintf(file, "%e\n", tsd.time);
        fprintf(file, "%e\n", tsd.l2efield1);
        fprintf(file, "%e\n", tsd.l2efield2);
        fprintf(file, "%e\n", tsd.mass);
        fprintf(file, "%e\n", tsd.momentum1);
        fprintf(file, "%e\n", tsd.momentum2);
        fprintf(file, "%e\n", tsd.energy);
        fprintf(file, "%e\n", tsd.l2norm);
        fprintf(file, "%e\n", tsd.minECFL);
        fprintf(file, "%e\n", tsd.maxECFL);
        fprintf(file, "%e\n", tsd.meanECFL);
        fprintf(file, "%e\n", tsd.xCFL);
    }
    
    // Write iters
    for(unsigned int i = 0; i < g_efieldIters.size(); i++)
        fprintf(file, "%d\n", g_efieldIters[i]);
    for(unsigned int i = 0; i < g_sweepIters.size(); i++)
        fprintf(file, "%d\n", g_sweepIters[i]);
    for(unsigned int i = 0; i < g_eulerIters.size(); i++)
        fprintf(file, "%d\n", g_eulerIters[i]);
    
    // Close file
    fclose(file);
}



/*
    Output efield data for testing purposes.
*/
void outputEfield(const char *filename, const int n, const double *phi, 
                  const double *E1, const double *E2)
{
    FILE *file;
    
    // Open file
    file = fopen(filename, "w");
    
    // Write phi, E1, E2
    for(int i = 0; i < n; i++) {
        double E2ToPrint = (E2 == NULL) ? 0.0 : E2[i];
        fprintf(file, "%.16e %.16e %.16e\n", phi[i], E1[i], E2ToPrint);
    }
    
    // Close file
    fclose(file);
}




