doTest=OFF
interface=OFF
mkl=OFF

for arg in $@
do
	if [[ "test" == "$arg" ]]; then
		doTest=ON
	fi
	if [[ "interface" == "$arg" ]]; then
		interface=ON
	fi
	if [[ "mkl" == "$arg" ]]; then
		mkl=ON
	fi
done

NICE_BUILD_PATH=${NICE_HOME}/cpp/build
cd ${NICE_BUILD_PATH}
cmake -Denable-mkl=$mkl -Denable-test=$doTest -Denable-interface=$interface ..
make -j 16
