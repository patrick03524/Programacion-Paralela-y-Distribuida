if [ -z $CUDA_INSTALL_PATH ]; then
    echo "Error. Please ensure CUDA_INSTALL_PATH is set before sourcing"
fi

if [ ! -d "$CUDA_INSTALL_PATH" ]; then
    echo "Error. Please ensure that CUDA_INSTALL_PATH is valid!"
fi

CUDA_VERSION=10.1

export NVIDIA_COMPUTE_SDK_LOCATION=$CUDA_INSTALL_PATH/samples
export LD_LIBRARY_PATH=$CUDA_INSTALL_PATH/lib64:$LD_LIBRARY_PATH

export PATH=$CUDA_INSTALL_PATH/bin:$PATH
export CC=gcc
export CXX=g++
