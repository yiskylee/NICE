if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake .. #-DCMAKE_CXX_FLAGS="-I ~/NICE/cpp/build/kmlocal/src/kmlocal/src/"
