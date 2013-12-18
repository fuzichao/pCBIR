echo "compiling feature_extractor"
g++ -fopenmp -ggdb `pkg-config --cflags opencv` -o feature feature.cpp `pkg-config --libs opencv` 2>1 compile_warning
echo "compiling query_handler"
mpic++ -o cluster cluster.cpp 2>1 compile_warining

