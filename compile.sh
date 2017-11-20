SOURCES="src/sweep1d.cpp \
         src/sweep1d_fast.cpp \
         src/sweep2d.cpp \
         src/sweep2d_fast.cpp \
         src/global.cpp \
         src/output.cpp \
         src/efield.cpp \
         src/input_deck_reader.cpp \
         src/init.cpp \
         src/timestep.cpp \
         src/euler_step.cpp \
         src/euler_step_nonlinear.cpp \
         src/utils.cpp \
         src/main.cpp \
         src/timestep_data.cpp"

PETSC_DIR="/Users/ckgarrett/lib/petsc-3.6.3"
PETSC_LIB="-L$PETSC_DIR/lib -lpetsc"
PETSC_INC="-I$PETSC_DIR/include"
LAPACK_LIB="$PETSC_DIR/lib/libflapack.a \
            $PETSC_DIR/lib/libfblas.a"

rm *.x
if [ "$1" == "serial" ]; then
    g++ -O3 -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_serial.x
elif [ "$1" == "omp" ]; then
    g++ -O3 -fopenmp -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_omp.x
elif [ "$1" == "debug" ]; then
    g++ -g -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_debug.x
else
    echo "Target Unknown"
fi


#g++ -O3 -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_serial.x
#g++ -O3 -fopenmp -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_omp.x
#g++ -g -fopenmp -Wall -Wno-unknown-pragmas $SOURCES $PETSC_INC $PETSC_LIB $LAPACK_LIB -o solver_debug.x

