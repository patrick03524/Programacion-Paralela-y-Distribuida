#GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
#GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
#GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
#GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
#GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
#GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
#GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
#GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"
GENCODE_SM70 ?= -gencode=arch=compute_70,code=\"sm_70,compute_70\"

CUSRC =
BASEEXE = oop_ubench_COAL
# CUDA source files (compiled with cudacc)
EXECUTABLE = $(BINDIR)/$(BINSUBDIR)/$(BASEEXE)
CUSRC += oo-vfunc-complete.cu
OPTS = -O3
CUOPTS = $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70)
INC = -I$(CUDA_INSTALL_PATH)/include
LIBS = -lcudart
PTX_GEN =  ../../../../../util/ptx_transform_coal/

all:
	nvcc --keep  $(NVOPTS) $(OPTS) $(CUOPTS) $(CUSRC) $(INC) -o $(EXECUTABLE) $(LIBS) 
	$(PTX_GEN)/generator.py oo-vfunc-complete.ptx
	cp oo-vfunc-complete.ptx_coal oo-vfunc-complete.ptx
	sh dryrun.sh
	rm -f *cpp* *fatbin* *cudafe*  *cubin* *.o *.module_id *dlink*

clean:
	rm -f $(EXECUTABLE) *cpp* *fatbin* *cudafe* *ptx *cubin* *.o *.module_id *dlink*
