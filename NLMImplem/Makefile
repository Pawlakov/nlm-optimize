.PHONY: NLMBasic
NLMBasic: NLMBasic.so
	cp NLMBasic.so ../NLMBase
	cp NLMBasic.so ../NLMBase/bin/Release/netcoreapp3.1
	rm -f NLMBasic.o NLMBasic.so

NLMBasic.so: NLMBasic.o
	g++ -shared -fopenmp -o NLMBasic.so NLMBasic.o

NLMBasic.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMBasic.cpp

.PHONY: NLMPlain
NLMPlain: NLMPlain.so
	cp NLMPlain.so ../NLMBase
	cp NLMPlain.so ../NLMBase/bin/Release/netcoreapp3.1
	rm -f NLMPlain.o NLMPlain.so

NLMPlain.so: NLMPlain.o
	g++ -shared -fopenmp -o NLMPlain.so NLMPlain.o

NLMPlain.o: NLMPlain.cpp NLMPlain.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMPlain.cpp

.PHONY: NLMGpgpu0
NLMGpgpu0: NLMGpgpu0.so
	cp NLMGpgpu0.so ../NLMBase
	cp NLMGpgpu0.so ../NLMBase/bin/Release/netcoreapp3.1
	rm -f NLMGpgpu0.hip.o NLMGpgpu0.so

NLMGpgpu0.so: NLMGpgpu0.hip.o
	hipcc -shared -o NLMGpgpu0.so NLMGpgpu0.hip.o

NLMGpgpu0.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -Wall -Werror -fpic --stdlib=libstdc++ NLMGpgpu0.hip.cpp

.PHONY: NLMGpgpu1
NLMGpgpu1: NLMGpgpu1.so
	cp NLMGpgpu1.so ../NLMBase
	cp NLMGpgpu1.so ../NLMBase/bin/Release/netcoreapp3.1
	rm -f NLMGpgpu1.hip.o NLMGpgpu1.so

NLMGpgpu1.so: NLMGpgpu1.hip.o
	hipcc -shared -o NLMGpgpu1.so NLMGpgpu1.hip.o

NLMGpgpu1.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -Wall -Werror -fpic --stdlib=libstdc++ NLMGpgpu1.hip.cpp