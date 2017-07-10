if [ ! -d build ]; then
  sudo mkdir build
fi
cd build
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ ..
