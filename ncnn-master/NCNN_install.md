#一、设置 aarch64 编译器相关路径

2、解压缩上述两个压缩包，并将gcc编译器加入系统可执行程序搜索路径中
export PATH=$PATH:~/work/linaro/gcc-linaro-7.1.1-2017.08-x86_64_aarch64-linux-gnu/bin

修改为：
export PATH=$PATH:/media/jcq/study/CrossTools/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu/bin


3、根据网上提供的配置CMakelist进行交叉编译的方法，修改ncnn工程的主CMakelist.txt文件，在文件开头新增以下内容：
SET(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
SET(CMAKE_FIND_ROOT_PATH "~/work/linaro/sysroot-glibc-linaro-2.25-2017.08-aarch64-linux-gnu")
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

修改为：
SET(CMAKE_FIND_ROOT_PATH "/media/jcq/study/CrossTools/gcc-linaro-4.9-2016.02-x86_64_aarch64-linux-gnu")

