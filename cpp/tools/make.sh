cd ../build
cmake -Denable-mkl=ON -Denable-test=ON -Denable-interface=ON ..
make -j 16
