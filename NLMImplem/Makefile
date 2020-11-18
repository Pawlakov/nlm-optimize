NLMBasic.so: NLMBasic.o
	g++ -shared -fopenmp -o NLMBasic.so NLMBasic.o
	cp NLMBasic.so ../NLMBase
	cp NLMBasic.so ../NLMBase/bin/Release/netcoreapp3.1

NLMBasic.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMBasic.cpp

NLMGpgpu.so: NLMGpgpu.o
	g++ -shared -fopenmp -o NLMGpgpu.so NLMGpgpu.o
	cp NLMGpgpu.so ../NLMBase
	cp NLMGpgpu.so ../NLMBase/bin/Release/netcoreapp3.1

NLMGpgpu.o: NLMGpgpu.cpp NLMGpgpu.h
	g++ -c -Wall -Werror -fpic -fopenmp NLMGpgpu.cpp

.PHONY: clean
clean:
	rm -f NLMBasic.o NLMBasic.so
	rm -f NLMGpgpu.o NLMGpgpu.so