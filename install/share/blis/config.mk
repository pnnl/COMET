#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2022, Advanced Micro Devices, Inc.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   - Neither the name(s) of the copyright holder(s) nor the names of its
#     contributors may be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#

# Only include this block of code once
ifndef CONFIG_MK_INCLUDED
CONFIG_MK_INCLUDED := yes

# The version string. This could be the official string or a custom
# string forced at configure-time.
VERSION           := 0.9.0

# The shared library .so major and minor.build version numbers.
SO_MAJOR          := 4
SO_MINORB         := 0.0
SO_MMB            := $(SO_MAJOR).$(SO_MINORB)

# The name of the configuration family.
CONFIG_NAME       := haswell

# The list of sub-configurations associated with CONFIG_NAME. Each
# sub-configuration in CONFIG_LIST corresponds to a configuration
# sub-directory in the 'config' directory. See the 'config_registry'
# file for the full list of registered configurations.
CONFIG_LIST       := haswell

# This list of kernels needed for the configurations in CONFIG_LIST.
# Each item in this list corresponds to a sub-directory in the top-level
# 'kernels' directory. Oftentimes, this list is identical to CONFIG_LIST,
# but not always. For example, if configuration X and Y use the same
# kernel set X, and configuration W uses kernel set Q, and the CONFIG_LIST
# might contained "X Y Z W", then the KERNEL_LIST would contain "X Z Q".
KERNEL_LIST       := haswell zen

# This list contains some number of "kernel:config" pairs, where "config"
# specifies which configuration's compilation flags (CFLAGS) should be
# used to compile the source code for the kernel set named "kernel".
KCONFIG_MAP       := haswell:haswell zen:haswell

# The operating system name, which should be either 'Linux' or 'Darwin'.
OS_NAME           := Darwin

# Check for whether the operating system is Windows.
IS_WIN            := no
IS_MSVC           := no

# The directory path to the top level of the source distribution. When
# building in-tree, this path is ".". When building out-of-tree, this path
# is path used to identify the location of configure. We also allow the
# includer of config.mk to override this value by setting DIST_PATH prior
# to including this file. This override option is employed, for example,
# when common.mk (and therefore config.mk) is included by the Makefile
# local to the 'testsuite' directory, or the 'test' directory containing
# individual test drivers.
ifeq ($(strip $(DIST_PATH)),)
DIST_PATH         := .
endif

# The C compiler.
CC_VENDOR         := clang
CC                := gcc

# Important C compiler ranges.
GCC_OT_4_9_0      := no
GCC_OT_6_1_0      := no
GCC_OT_9_1_0      := no
GCC_OT_10_3_0     := no
CLANG_OT_9_0_0    := no
CLANG_OT_12_0_0   := no
AOCC_OT_2_0_0     := no
AOCC_OT_3_0_0     := no

# The C++ compiler. NOTE: A C++ is typically not needed.
CXX               := g++

# Static library indexer.
RANLIB            := ranlib

# Archiver.
AR                := ar

# Python Interpreter
PYTHON            := python3

# Preset (required) CFLAGS and LDFLAGS. These variables capture the value
# of the CFLAGS and LDFLAGS environment variables at configure-time (and/or
# the value of CFLAGS/LDFLAGS if either was specified on the command line).
# These flags are used in addition to the flags automatically determined
# by the build system.
CFLAGS_PRESET     := 
LDFLAGS_PRESET    := 

# The level of debugging info to generate.
DEBUG_TYPE        := off
ENABLE_DEBUG      := no

# Whether to compile and link the AddressSanitizer library.
MK_ENABLE_ASAN    := no

# Whether operating system support was requested via --enable-system.
ENABLE_SYSTEM     := yes

# The requested threading model(s).
THREADING_MODEL   := single

# Whether the compiler supports "#pragma omp simd" via the -fopenmp-simd option.
PRAGMA_OMP_SIMD   := yes

# The installation prefix, exec_prefix, libdir, includedir, and shareddir
# values from configure tell us where to install the libraries, header files,
# and public makefile fragments. We must first assign each substituted
# @anchor@ to its own variable. Why? Because the subsitutions may contain
# unevaluated variable expressions. For example, '${exec_prefix}/lib' may be replaced
# with '${exec_prefix}/lib'. By assigning the anchors to variables first, and
# then assigning them to their final INSTALL_* variables, we allow prefix and
# exec_prefix to be used in the definitions of exec_prefix, libdir,
# includedir, and sharedir.
prefix              := /Users/flyn908/Projects/COMET/install
exec_prefix         := ${prefix}
libdir              := ${exec_prefix}/lib
includedir          := ${prefix}/include
sharedir            := ${prefix}/share

# Notice that we support the use of DESTDIR so that advanced users may install
# to a temporary location.
INSTALL_LIBDIR      := $(DESTDIR)$(libdir)
INSTALL_INCDIR      := $(DESTDIR)$(includedir)
INSTALL_SHAREDIR    := $(DESTDIR)$(sharedir)

#$(info prefix      = $(prefix) )
#$(info exec_prefix = $(exec_prefix) )
#$(info libdir      = $(libdir) )
#$(info includedir  = $(includedir) )
#$(info sharedir    = $(sharedir) )
#$(error .)

# Whether to output verbose command-line feedback as the Makefile is
# processed.
ENABLE_VERBOSE    := no

# Whether we are building out-of-tree.
BUILDING_OOT      := no

# Whether we need to employ an alternate method for passing object files to
# ar and/or the linker to work around a small value of ARG_MAX.
ARG_MAX_HACK      := no

# Whether to build the static and shared libraries.
# NOTE: The "MK_" prefix, which helps differentiate these variables from
# their corresonding cpp macros that use the BLIS_ prefix.
MK_ENABLE_STATIC  := yes
MK_ENABLE_SHARED  := yes

# Whether to use an install_name based on @rpath.
MK_ENABLE_RPATH   := no

# Whether to export all symbols within the shared library, even those symbols
# that are considered to be for internal use only.
EXPORT_SHARED     := public

# Whether to enable either the BLAS or CBLAS compatibility layers.
MK_ENABLE_BLAS    := yes
MK_ENABLE_CBLAS   := no

# Whether libblis will depend on libmemkind for certain memory allocations.
MK_ENABLE_MEMKIND := no

# The names of the addons to include when building BLIS. If empty, no addons
# will be included.
ADDON_LIST        := 

# The name of a sandbox defining an alternative gemm implementation. If empty,
# no sandbox will be used and the conventional gemm implementation will remain
# enabled.
SANDBOX           := 

# The name of the pthreads library. If --disable-system was given, then this
# variable is set to the empty value.
LIBPTHREAD        := -lpthread

# Whether we should use AMD-customized versions of certain framework files.
ENABLE_AMD_FRAME_TWEAKS := no

# end of ifndef CONFIG_MK_INCLUDED conditional block
endif
