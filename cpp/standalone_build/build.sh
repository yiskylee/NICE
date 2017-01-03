cd ../build
cmake -Denable-test=OFF ..
make -j 16
cd -
#nvcc -arch=sm_50 --std=c++11 -I../build/eigen/src/eigen -I../ -I../build/gtest/src/googletest/googletest/include/ -L/usr/local/cuda-7.0/lib64 -lcudart -lcudadevrt -lcublas_device -L../dlib/build -ldlib -L../build/gtest/src/googletest-build/googlemock/gtest -lgtest -lgtest_main ../test/model_test/kdac_test.cc
nvcc ../test/model_test/kdac_test.cc ../src/cuda/kdac_cuda.cu ../src/util/gpu_util.cc -arch=sm_50 --std=c++11 -I../build/eigen/src/eigen -I../ -I../build/gtest/src/googletest/googletest/include/ -L/usr/local/cuda-7.0/lib64 -lcudart -lcublas_device -lcudadevrt -lcublas -lcusolver -L../dlib/build -ldlib -L../build/gtest/src/googletest-build/googlemock/gtest -lgtest -lgtest_main
