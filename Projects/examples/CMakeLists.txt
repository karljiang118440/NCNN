

cmake_minimum_required(VERSION 2.8) # cmake verson，指定cmake版本 
add_definitions(-std=c++11 -g)

project (examples)


set(ncnn_test_INCLUDE_DIRS /media/jcq/Soft/NCNN/ncnn/build-host-gcc-linux/install/include/ncnn)

include_directories(
	${ncnn_test_INCLUDE_DIRS} 

)

### OpenCV的配置 ###
#set (OpenCV_DIR /usr) 

#set (OpenCV_DIR /media/jcq/Soft/Opencv/opencv-3.4.2/build)
#find_package(OpenCV 3 REQUIRED)



set (OpenCV_DIR /media/jcq/Soft/Opencv/opencv-4.1.2/build)
find_package(OpenCV 4 REQUIRED)

add_executable (example mobilenetssd.cpp)


set (NCNN_LIBS /media/jcq/Soft/NCNN/ncnn/build-host-gcc-linux/install/lib/libncnn.a)
set (GOMP_LIB /usr/lib/x86_64-linux-gnu/libgomp.so.1)

target_link_libraries( example
	${OpenCV_LIBS} 
	${NCNN_LIBS} 
	${GOMP_LIB}
)


