NLMBasic.so: NLMBasic.o
	g++ -shared -fopenmp -o NLMBasic.so NLMBasic.o
	cp NLMBasic.so ../NLMBase
	cp NLMBasic.so ../NLMBase/bin/Release/netcoreapp3.1

NLMBasic.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMBasic.cpp

NLMGpgpu0.so: NLMGpgpu0.hip.o
	hipcc -shared -o NLMGpgpu0.so NLMGpgpu0.hip.o
	cp NLMGpgpu0.so ../NLMBase
	cp NLMGpgpu0.so ../NLMBase/bin/Release/netcoreapp3.1

NLMGpgpu0.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -Wall -Werror -fpic --stdlib=libstdc++ NLMGpgpu0.hip.cpp

NLMGpgpu1.so: NLMGpgpu1.hip.o
	hipcc -shared -o NLMGpgpu1.so NLMGpgpu1.hip.o
	cp NLMGpgpu1.so ../NLMBase
	cp NLMGpgpu1.so ../NLMBase/bin/Release/netcoreapp3.1

NLMGpgpu1.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -Wall -Werror -fpic --stdlib=libstdc++ NLMGpgpu1.hip.cpp

.PHONY: clean
clean:
	rm -f NLMBasic.o NLMBasic.so
	rm -f NLMGpgpu0.hip.o NLMGpgpu0.so
	rm -f NLMGpgpu1.hip.o NLMGpgpu1.so