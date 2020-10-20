##############################################################################
#
# Freescale Confidential Proprietary
#
# Copyright (c) 2016 Freescale Semiconductor;
# All Rights Reserved
#
##############################################################################
#
# THIS SOFTWARE IS PROVIDED BY FREESCALE "AS IS" AND ANY EXPRESSED OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL FREESCALE OR ITS CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
##############################################################################

SDK_ROOT := $(call path_relative_to,$(CURDIR),$(CURR_SDK_ROOT))

##############################################################################
# ARM_APP
##############################################################################

ARM_APP = yolo-fastest

ARM_APP_SRCS +=                                                               \
    yolo-fastest.cpp                                                                 \

ARM_INCS =                                                                   \
    -I$(SDK_ROOT)/3rdparty/ncnn/include/ncnn   \

##############################################################################
# STANDALONE SPECIFIC INCLUDES
##############################################################################	
ifneq (,$(findstring -sa,$(ODIR)))

ARM_APP_SYSTEM_LIBS +=                                                       \
    $(SDK_ROOT)/libs/startup/v234ce_standalone/$(ODIR)/libv234ce.a           \
    $(SDK_ROOT)/libs/io/i2c/$(ODIR)/libi2c.a                                 \
    $(SDK_ROOT)/libs/io/uartlinflex_io/$(ODIR)/liblinflex.a   \
    $(SDK_ROOT)/3rdparty/ncnn/lib/libncnn.a          \                
    
endif

ARM_INCS +=                                                                              \
    -I$(SDK_ROOT)/demos/airunner/libs/utils/include                                      \
    -I$(SDK_ROOT)/include                                                                \
    -I$(SDK_ROOT)/libs/apex/acf/include                                                  \
    -I$(SDK_ROOT)/libs/apex/drivers/user/include                                         \
    -I$(SDK_ROOT)/libs/apex/icp/include                                                  \
    -I$(SDK_ROOT)/libs/dnn/airunner/postprocessing/include                               \
    -I$(SDK_ROOT)/libs/dnn/airunner/preprocessing/include                                \
    -I$(SDK_ROOT)/libs/io/dcu/include                                                    \
    -I$(SDK_ROOT)/libs/io/frame_io/include                                               \
    -I$(SDK_ROOT)/libs/utils/common/include                                              \
    -I$(OPENCV_ROOT)/include                                                             \
    -I$(SDK_ROOT)/3rdparty/protobuf/include

ARM_APP_LIBS =                                                                           \
    $(SDK_ROOT)/demos/airunner/libs/utils/$(ODIR)/demo_utils.a                           \
    $(SDK_ROOT)/libs/dnn/airunner/core/$(ODIR)/airunner.a                                \
    $(SDK_ROOT)/libs/dnn/airunner/importer/$(ODIR)/airunner_importer.a                   \
    $(SDK_ROOT)/libs/dnn/airunner/nodes/apu2/$(ODIR)/airunner_apu2_nodes.a               \
    $(SDK_ROOT)/libs/dnn/airunner/nodes/cpu_opt/$(ODIR)/airunner_cpu_opt_nodes.a         \
    $(SDK_ROOT)/libs/dnn/airunner/preprocessing/$(ODIR)/airunner_preprocessing.a         \
    $(SDK_ROOT)/libs/dnn/airunner/postprocessing/$(ODIR)/airunner_postprocessing.a       \
    $(SDK_ROOT)/libs/apex/acf/$(ODIR)/libacf.a                                           \
    $(SDK_ROOT)/libs/apex/icp/$(ODIR)/libicp.a                                           \
    $(SDK_ROOT)/libs/io/frame_io/$(ODIR)/libframe_io.a                                   \
    $(SDK_ROOT)/libs/utils/communications/$(ODIR)/lib_communications.a                   \
    $(SDK_ROOT)/libs/apex/drivers/user/$(ODIR)/libapexdrv.a                              \
    $(SDK_ROOT)/libs/utils/log/$(ODIR)/liblog.a                                           \
    $(SDK_ROOT)/3rdparty/ncnn/lib/libncnn.a          \

ARM_APP_LIBS +=                                                                          \
    $(SDK_ROOT)/libs/utils/common/$(ODIR)/libcommon.a                                    \
    $(SDK_ROOT)/libs/utils/umat/$(ODIR)/libumat.a                                        \
    $(SDK_ROOT)/libs/utils/sumat/$(ODIR)/libsumat.a                                      \
    $(SDK_ROOT)/libs/io/semihost/$(ODIR)/libSemihost.a

ARM_LDOPTS +=                                                                            \
	  -L$(SDK_ROOT)/3rdparty/tensorflow/$(EXTODIR)                                         \
	  -L$(SDK_ROOT)/3rdparty/protobuf/$(EXTODIR)                                           \
	  -lprotobuf-lite                                                                      \
	  -l3rdparty_tensorflow                                                                \
    -lopencv_core                                                                        \
    -lopencv_imgproc                                                                     \
    -lopencv_imgcodecs

##############################################################################
# STANDALONE SPECIFIC INCLUDES
##############################################################################
ifneq (,$(findstring -sa,$(ODIR)))

ARM_APP_LIBS +=                                                                          \
    $(SDK_ROOT)/libs/startup/v234ce_standalone/$(ODIR)/libv234ce.a                       \
    $(SDK_ROOT)/libs/io/i2c/$(ODIR)/libi2c.a                                             \
    $(SDK_ROOT)/libs/io/semihost/$(ODIR)/libSemihost.a                                   \
    $(SDK_ROOT)/libs/io/uartlinflex_io/$(ODIR)/liblinflex.a

ARM_LDOPTS += -lzlib

endif

##############################################################################
# INTEGRITY SPECIFIC INCLUDES
##############################################################################
ifneq (,$(findstring -integrity,$(ODIR)))

ARM_LDOPTS +=                                                                \
    -L$(SDK_ROOT)/ocv/integrity-arm/share/OpenCV/3rdparty/lib                \
    -lopencv_core                                                            \
    -lposix                                                                  \
    -livfs                                                                   \
    -lIlmImf                                                                 \
    -lzlib                                                                   \
    --exceptions

endif

##############################################################################
# X86 TARGET VARIABLES
##############################################################################
ifneq (,$(findstring x86-gnu,$(ODIR)))

X86_APP = $(ARM_APP)

X86_APP_SRCS += $(ARM_APP_SRCS)

X86_INCS += $(ARM_INCS)

X86_APP_LIBS +=                                                                         \
    $(SDK_ROOT)/3rdparty/protobuf/win32-mingw/lib/libprotobuf.a                         \
    $(SDK_ROOT)/demos/airunner/libs/utils/$(ODIR)/demo_utils.a                          \
    $(SDK_ROOT)/libs/dnn/airunner/core/$(ODIR)/airunner.a                               \
    $(SDK_ROOT)/libs/dnn/airunner/importer/$(ODIR)/airunner_importer.a                  \
    $(SDK_ROOT)/libs/dnn/airunner/preprocessing/$(ODIR)/airunner_preprocessing.a        \
    $(SDK_ROOT)/libs/dnn/airunner/postprocessing/$(ODIR)/airunner_postprocessing.a      \
    $(SDK_ROOT)/libs/dnn/airunner/nodes/cpu_opt/$(ODIR)/airunner_cpu_opt_nodes.a        \
    $(SDK_ROOT)/3rdparty/tensorflow/win32-mingw/lib/lib3rdparty_tensorflow.a            \
    $(SDK_ROOT)/3rdparty/ocv/win32-mingw/x86/mingw/lib/libopencv_core310.dll.a          \
    $(SDK_ROOT)/3rdparty/ocv/win32-mingw/x86/mingw/lib/libopencv_imgproc310.dll.a       \
    $(SDK_ROOT)/3rdparty/ocv/win32-mingw/x86/mingw/lib/libopencv_imgcodecs310.dll.a     \


endif


##############################################################################
# QNX SPECIFIC INCLUDES
##############################################################################
ifneq (,$(findstring qcc-qnx,$(ODIR)))

ARM_LDOPTS +=                                                                \
    -lopencv_imgproc                                                         \
    -lopencv_imgcodecs                                                       \
    -lscreen

endif
