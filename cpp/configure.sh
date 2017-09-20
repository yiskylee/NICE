if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake -DMKLROOT=/opt/intel/compilers_and_libraries_2016.2.181/ -Denable-mkl=ON -Denable-test=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ ..
