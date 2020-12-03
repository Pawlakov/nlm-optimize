NLMBasic.so: NLMBasic.o
	g++ -shared -fopenmp -o NLMBasic.so NLMBasic.o
	cp NLMBasic.so ../NLMBase
	cp NLMBasic.so ../NLMBase/bin/Release/netcoreapp3.1

NLMBasic.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMBasic.cpp

NLMGpgpu.so: NLMGpgpu.hip.o
	hipcc -shared -o NLMGpgpu.so NLMGpgpu.hip.o
	cp NLMGpgpu.so ../NLMBase
	cp NLMGpgpu.so ../NLMBase/bin/Release/netcoreapp3.1

NLMGpgpu.hip.o: NLMGpgpu.hip.cpp NLMGpgpu.hip.h
	hipcc -c -Wall -Werror -fpic --stdlib=libstdc++ NLMGpgpu.hip.cpp

.PHONY: clean
clean:
	rm -f NLMBasic.o NLMBasic.so
	rm -f NLMGpgpu.hip.o NLMGpgpu.so