if [ ! -d build ]; then
  mkdir build
fi
cd build
cmake -Denable-intelmkl=OFF -Denable-condamkl=ON -Denable-test=ON -Denable-interface=ON -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ ..
cp ../tools/make.sh ./
