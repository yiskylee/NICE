doTest=OFF
interface=OFF
intemkl=OFF
condamkl=ON

for arg in $@
do
	if [[ "test" == "$arg" ]]; then
		doTest=ON
	fi
	if [[ "interface" == "$arg" ]]; then
		interface=ON
	fi
	if [[ "intelmkl" == "$arg" ]]; then
		intelmkl=ON
	fi
	if [[ "condamkl" == "$arg" ]]; then
		condamkl=ON
	fi
done

NICE_BUILD_PATH=${NICE_HOME}/cpp/build
cd ${NICE_BUILD_PATH}
rm CMakeCache.txt
cmake -Denable-condamkl=$condamkl -Denable-intelmkl=$intelmkl -Denable-test=$doTest -Denable-interface=$interface ..
echo "cmake -Denable-condamkl=$condamkl -Denable-intelmkl=$intelmkl -Denable-test=$doTest -Denable-interface=$interface .."
make -j 16
