.PHONY: all
all: NLMPara NLMParaOpti NLM NLMOpti NLMGpgpu0 NLMGpgpu0Opti NLMGpgpu1 NLMGpgpu1Opti

.PHONY: NLMPara
NLMPara: NLMPara.so
	mkdir -p bin
	cp NLMPara.so ./bin
	rm -f NLMPara.o NLMPara.so

NLMPara.so: NLMPara.o
	g++ -shared -fopenmp -o NLMPara.so NLMPara.o

NLMPara.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Wextra -fpic -fopenmp -o NLMPara.o NLMBasic.cpp

.PHONY: NLMParaOpti
NLMParaOpti: NLMParaOpti.so
	mkdir -p bin
	cp NLMParaOpti.so ./bin
	rm -f NLMParaOpti.o NLMParaOpti.so

NLMParaOpti.so: NLMParaOpti.o
	g++ -shared -fopenmp -o NLMParaOpti.so NLMParaOpti.o

NLMParaOpti.o: NLMBasic.cpp NLMBasic.h
	g++ -c -O3 -funroll-loops -fomit-frame-pointer -fno-tree-pre -falign-loops -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -ansi -fpic -fopenmp -o NLMParaOpti.o NLMBasic.cpp

.PHONY: NLM
NLM: NLM.so
	mkdir -p bin
	cp NLM.so ./bin
	rm -f NLM.o NLM.so

NLM.so: NLM.o
	g++ -shared -o NLM.so NLM.o

NLM.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Wextra -fpic -o NLM.o NLMBasic.cpp

.PHONY: NLMOpti
NLMOpti: NLMOpti.so
	mkdir -p bin
	cp NLMOpti.so ./bin
	rm -f NLMOpti.o NLMOpti.so

NLMOpti.so: NLMOpti.o
	g++ -shared -o NLMOpti.so NLMOpti.o

NLMOpti.o: NLMBasic.cpp NLMBasic.h
	g++ -c -O3 -funroll-loops -fomit-frame-pointer -fno-tree-pre -falign-loops -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -ansi -fpic -o NLMOpti.o NLMBasic.cpp

.PHONY: NLMGpgpu0
NLMGpgpu0: NLMGpgpu0.so
	mkdir -p bin
	cp NLMGpgpu0.so ./bin
	rm -f NLMGpgpu0.hip.o NLMGpgpu0.so

NLMGpgpu0.so: NLMGpgpu0.hip.o
	hipcc -shared -o NLMGpgpu0.so NLMGpgpu0.hip.o

NLMGpgpu0.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -Wall -Wextra -fpic --stdlib=libstdc++ NLMGpgpu0.hip.cpp

.PHONY: NLMGpgpu0Opti
NLMGpgpu0Opti: NLMGpgpu0Opti.so
	mkdir -p bin
	cp NLMGpgpu0Opti.so ./bin
	rm -f NLMGpgpu0Opti.hip.o NLMGpgpu0Opti.so

NLMGpgpu0Opti.so: NLMGpgpu0Opti.hip.o
	hipcc -shared -o NLMGpgpu0Opti.so NLMGpgpu0Opti.hip.o

NLMGpgpu0Opti.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -O3 -funroll-loops -fomit-frame-pointer -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -fpic --stdlib=libstdc++ -o NLMGpgpu0Opti.hip.o NLMGpgpu0.hip.cpp

.PHONY: NLMGpgpu1
NLMGpgpu1: NLMGpgpu1.so
	mkdir -p bin
	cp NLMGpgpu1.so ./bin
	rm -f NLMGpgpu1.hip.o NLMGpgpu1.so

NLMGpgpu1.so: NLMGpgpu1.hip.o
	hipcc -shared -o NLMGpgpu1.so NLMGpgpu1.hip.o

NLMGpgpu1.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -Wall -Wextra -fpic --stdlib=libstdc++ NLMGpgpu1.hip.cpp
	
.PHONY: NLMGpgpu1Opti
NLMGpgpu1Opti: NLMGpgpu1Opti.so
	mkdir -p bin
	cp NLMGpgpu1Opti.so ./bin
	rm -f NLMGpgpu1Opti.hip.o NLMGpgpu1Opti.so

NLMGpgpu1Opti.so: NLMGpgpu1Opti.hip.o
	hipcc -shared -o NLMGpgpu1Opti.so NLMGpgpu1Opti.hip.o

NLMGpgpu1Opti.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -O3 -funroll-loops -fomit-frame-pointer -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -fpic --stdlib=libstdc++ -o NLMGpgpu1Opti.hip.o NLMGpgpu1.hip.cpp