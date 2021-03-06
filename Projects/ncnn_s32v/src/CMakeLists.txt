

set(ncnn_test_INCLUDE_DIRS /media/jcq/Soft/NCNN/ncnn/build-host-gcc-linux/install/include/ncnn)


macro(ncnn_add_example name)
    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -std=c++11)
    target_link_libraries(${name} PRIVATE ncnn ${OpenCV_LIBS})

    # add test to a virtual project group
    set_property(TARGET ${name} PROPERTY FOLDER "examples")
endmacro()

find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc)
endif()

if(OpenCV_FOUND)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
    include_directories(${CMAKE_CURRENT_BINARY_DIR}/../src)

   # ncnn_add_example(squeezenet)
   # ncnn_add_example(fasterrcnn)
   # ncnn_add_example(rfcn)
    #ncnn_add_example(yolov2)
    #ncnn_add_example(yolov3)
    #ncnn_add_example(mobilenetv2ssdlite)
    #ncnn_add_example(mobilenetssd)
    #ncnn_add_example(mobilenetssd-0107)
    ncnn_add_example(yoloface-500k-lanmark106)
   # ncnn_add_example(squeezenetssd)
    #ncnn_add_example(shufflenetv2)
    #ncnn_add_example(peleenetssd_seg)
    #ncnn_add_example(simplepose)
    #ncnn_add_example(retinaface)
    #ncnn_add_example(yolact)
else()
    message(WARNING "OpenCV not found, examples won't be built")
endif()
