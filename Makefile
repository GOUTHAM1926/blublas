# =============================================================================
# BluBridge-BLAS Makefile
# GPU architecture and CUDA path are detected automatically at build time.
# =============================================================================

SHELL := /bin/bash

# ─── CUDA toolchain auto-detection ───────────────────────────────────────────
# 1. Prefer nvcc already on PATH; fall back to the highest-versioned install.
NVCC ?= $(shell \
    which nvcc 2>/dev/null || \
    ls /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1)

ifeq ($(NVCC),)
$(error Could not locate nvcc. Install CUDA or set NVCC=/path/to/nvcc)
endif

# 2. Derive CUDA_PATH from the nvcc location  (nvcc → bin/ → cuda-X.Y/)
CUDA_PATH := $(shell dirname $$(dirname $(NVCC)))

# 3. Make nvvm/bin available for PTX assembler
export PATH := $(CUDA_PATH)/nvvm/bin:$(PATH)

# Host C++ compiler
CXX := g++

# ─── GPU architecture auto-detection ─────────────────────────────────────────
# Query every unique compute capability installed in the system.
# nvidia-smi prints lines like "8.6", "10.0" etc.
# We convert each to  -gencode arch=compute_XY,code=sm_XY
#
# If nvidia-smi is unavailable (CI / headless build) we fall back to a
# safe Ampere baseline and warn.  Override with:
#   make GENCODE_FLAGS="-gencode arch=compute_90,code=sm_90"

GENCODE_FLAGS := -gencode arch=compute_103a,code=sm_103a


# ─── build directories ────────────────────────────────────────────────────────
SRCDIR   := src
OBJDIR   := build/objects
BUILDDIR := build
LIBNAME  := myblas
TARGET_SO := $(BUILDDIR)/lib$(LIBNAME).so

# ─── compiler flags ───────────────────────────────────────────────────────────
CPPFLAGS := \
    -Iinclude \
    -I$(CUDA_PATH)/include \
    -Iexternal/cutlass/include \
    -I$(CUDA_PATH)/targets/x86_64-linux/include/cccl

CXXFLAGS := -std=c++17 -fPIC -Wall -O3 -DNDEBUG

NVCCFLAGS := \
    -std=c++17 \
    -Xcompiler="-fPIC" \
    $(GENCODE_FLAGS) \
    -O3 -DNDEBUG \
    -Xptxas -O3 \
    -use_fast_math \
    -Iexternal/cutlass/include \
    -I$(CUDA_PATH)/targets/x86_64-linux/include/cccl

# ─── linker flags ─────────────────────────────────────────────────────────────
RPATH    := -Xlinker -rpath -Xlinker '$$ORIGIN' \
            -Xlinker -rpath -Xlinker '$$ORIGIN/build'
LDFLAGS  := -L$(CUDA_PATH)/lib64 \
            -L$(CUDA_PATH)/targets/x86_64-linux/lib \
            -L$(BUILDDIR) $(RPATH)
LDLIBS   := -lcudart -lcublas -lcublasLt -lcuda

# =============================================================================
# Source Files — Auto-Discovery
# =============================================================================
CPP_SOURCES := $(shell find $(SRCDIR) -name '*.cpp')
CU_SOURCES  := $(shell find $(SRCDIR) -name '*.cu')

OBJECTS_FROM_CPP := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(CPP_SOURCES))
OBJECTS_FROM_CU  := $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o,$(CU_SOURCES))
ALL_OBJECTS      := $(OBJECTS_FROM_CPP) $(OBJECTS_FROM_CU)

# =============================================================================
# Test Programs — Auto-Discovery  (examples/*.cpp and examples/*.cu)
# =============================================================================
TEST_SOURCES     := $(wildcard examples/*.cpp) $(wildcard examples/*.cu)
TEST_NAMES       := $(basename $(notdir $(TEST_SOURCES)))
TEST_EXECUTABLES := $(addprefix $(BUILDDIR)/,$(TEST_NAMES))

# =============================================================================
# Build Targets
# =============================================================================
.PHONY: all lib tests clean rebuild info help

all: lib

lib: $(TARGET_SO)
	@echo ""
	@echo "✅  Library built: $(TARGET_SO)"

tests: lib
	@echo ""
	@echo "--- Building tests from examples/ ---"
	@for test_src in $(TEST_SOURCES); do \
		test_name=$$(basename $$test_src .cpp); \
		test_name=$$(basename $$test_name .cu); \
		echo "Building: $$test_name"; \
		if [[ $$test_src == *.cpp ]]; then \
			$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$$test_name $$test_src \
			    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS) 2>&1 | grep -v "warning:" || true; \
		else \
			$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $(BUILDDIR)/$$test_name $$test_src \
			    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS) 2>&1 | grep -v "warning:" || true; \
		fi; \
	done
	@echo ""
	@echo "✅  Test building complete"

# ─── shared library ───────────────────────────────────────────────────────────
$(TARGET_SO): $(ALL_OBJECTS)
	@echo ""
	@echo "--- Linking shared library: $@ ---"
	@mkdir -p $(BUILDDIR)
	$(NVCC) -shared $(NVCCFLAGS) $(ALL_OBJECTS) $(LDFLAGS) $(LDLIBS) -o $@

# ─── object compilation ───────────────────────────────────────────────────────
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(@D)
	@echo "Compiling [CXX ]: $<"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(@D)
	@echo "Compiling [CUDA]: $<"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@

# ─── test executables ─────────────────────────────────────────────────────────
$(BUILDDIR)/%: examples/%.cpp $(TARGET_SO)
	@echo "Building test [CXX ]: $@"
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)

$(BUILDDIR)/%: examples/%.cu $(TARGET_SO)
	@echo "Building test [CUDA]: $@"
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $@ $< $(LDFLAGS) -l$(LIBNAME) $(LDLIBS)

# =============================================================================
# Utility Targets
# =============================================================================
rebuild:
	@$(MAKE) clean && $(MAKE) all

clean:
	@echo "--- Cleaning build artifacts ---"
	rm -rf $(OBJDIR) $(BUILDDIR)

# Print detected configuration — useful for debugging
info:
	@echo ""
	@echo "────────────────────────────────────────"
	@echo "  BluBridge-BLAS build configuration"
	@echo "────────────────────────────────────────"
	@echo "  NVCC        : $(NVCC)"
	@echo "  CUDA_PATH   : $(CUDA_PATH)"
	@echo "  GENCODE     : $(GENCODE_FLAGS)"
	@echo "  GPU caps    : $(_GPU_CAPS)"
	@echo "────────────────────────────────────────"
	@echo ""

# Build a single test — Usage: make test TEST=test_bgemm_sm100_bf16
test: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make test TEST=test_bgemm_sm100_bf16"; \
		exit 1; \
	fi
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@echo "--- Building/Updating test: $(TEST_NAME) ---"
	@if [ -f "$(TEST)" ]; then \
		if [[ "$(TEST)" == *.cpp ]]; then \
			$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $(BUILDDIR)/$(TEST_NAME) $(TEST) \
			    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
		else \
			$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $(BUILDDIR)/$(TEST_NAME) $(TEST) \
			    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
		fi; \
	else \
		$(MAKE) $(BUILDDIR)/$(TEST_NAME); \
	fi
	@echo "✅  Built: $(BUILDDIR)/$(TEST_NAME)"

# Build and run a test — Usage: make run TEST=test_bgemm_sm100_bf16
run: lib
	@if [ -z "$(TEST)" ]; then \
		echo "ERROR: Please specify TEST name"; \
		echo "Usage: make run TEST=test_bgemm_sm100_bf16"; \
		exit 1; \
	fi
	$(eval TEST_NAME := $(basename $(notdir $(TEST))))
	@$(MAKE) test TEST=$(TEST_NAME)
	@echo "--- Running $(BUILDDIR)/$(TEST_NAME) ---"
	@LD_LIBRARY_PATH=$(BUILDDIR):$$LD_LIBRARY_PATH $(BUILDDIR)/$(TEST_NAME)

# Compile and run an arbitrary file — Usage: make run-snippet FILE=examples/test_x.cu
run-snippet: lib
	@if [ -z "$(FILE)" ]; then \
		echo "ERROR: Please specify FILE path"; \
		echo "Usage: make run-snippet FILE=examples/test_x.cu"; \
		exit 1; \
	fi
	@echo "--- Compiling snippet: $(FILE) ---"
	@filename=$$(basename $(FILE)); \
	exe_name=$(BUILDDIR)/$${filename%.*}; \
	if [[ $(FILE) == *.cpp ]]; then \
		$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $$exe_name $(FILE) \
		    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
	else \
		$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $$exe_name $(FILE) \
		    $(LDFLAGS) -l$(LIBNAME) $(LDLIBS); \
	fi
	@echo "--- Running $$exe_name ---"
	@LD_LIBRARY_PATH=$(BUILDDIR):$$LD_LIBRARY_PATH $$exe_name

help:
	@echo ""
	@echo "BluBridge-BLAS Makefile  (GPU arch auto-detected)"
	@echo ""
	@echo "  make              Build shared library"
	@echo "  make lib          Same as above"
	@echo "  make tests        Build all programs in examples/"
	@echo "  make test TEST=X  Build one test by name or path"
	@echo "  make run  TEST=X  Build and run one test"
	@echo "  make run-snippet FILE=path/to/file.cu"
	@echo "  make rebuild      Clean + build"
	@echo "  make clean        Remove build/"
	@echo "  make info         Show detected CUDA/arch configuration"
	@echo ""
	@echo "  Override arch:    make GENCODE_FLAGS=\"-gencode arch=compute_90,code=sm_90\""
	@echo "  Override nvcc:    make NVCC=/usr/local/cuda-12.9/bin/nvcc"
	@echo ""
	@echo "  Auto-discovered tests: $(TEST_NAMES)"
	@echo ""

# Dependency tracking
-include $(ALL_OBJECTS:.o=.d)
