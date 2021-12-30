.PHONY: NLMBasic
NLMBasic: NLMBasic.so
	cp NLMBasic.so ../NLMBase
	cp NLMBasic.so ../NLMBase/bin/Release/net5.0
	rm -f NLMBasic.o NLMBasic.so

NLMBasic.so: NLMBasic.o
	g++ -shared -fopenmp -o NLMBasic.so NLMBasic.o

NLMBasic.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Wextra -fpic -fopenmp NLMBasic.cpp

.PHONY: NLMBasicOpti
NLMBasicOpti: NLMBasicOpti.so
	cp NLMBasicOpti.so ../NLMBase
	cp NLMBasicOpti.so ../NLMBase/bin/Release/net5.0
	rm -f NLMBasicOpti.o NLMBasicOpti.so

NLMBasicOpti.so: NLMBasicOpti.o
	g++ -shared -fopenmp -o NLMBasicOpti.so NLMBasicOpti.o

NLMBasicOpti.o: NLMBasic.cpp NLMBasic.h
	g++ -c -O3 -funroll-loops -fomit-frame-pointer -fno-tree-pre -falign-loops -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -ansi -fpic -fopenmp -o NLMBasicOpti.o NLMBasic.cpp

.PHONY: NLMPlain
NLMPlain: NLMPlain.so
	cp NLMPlain.so ../NLMBase
	cp NLMPlain.so ../NLMBase/bin/Release/net5.0
	rm -f NLMPlain.o NLMPlain.so

NLMPlain.so: NLMPlain.o
	g++ -shared -o NLMPlain.so NLMPlain.o

NLMPlain.o: NLMBasic.cpp NLMBasic.h
	g++ -c -Wall -Wextra -fpic -o NLMPlain.o NLMBasic.cpp

.PHONY: NLMPlainOpti
NLMPlainOpti: NLMPlainOpti.so
	cp NLMPlainOpti.so ../NLMBase
	cp NLMPlainOpti.so ../NLMBase/bin/Release/net5.0
	rm -f NLMPlainOpti.o NLMPlainOpti.so

NLMPlainOpti.so: NLMPlainOpti.o
	g++ -shared -o NLMPlainOpti.so NLMPlainOpti.o

NLMPlainOpti.o: NLMBasic.cpp NLMBasic.h
	g++ -c -O3 -funroll-loops -fomit-frame-pointer -fno-tree-pre -falign-loops -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -ansi -fpic -o NLMPlainOpti.o NLMBasic.cpp

.PHONY: NLMGpgpu0
NLMGpgpu0: NLMGpgpu0.so
	cp NLMGpgpu0.so ../NLMBase
	cp NLMGpgpu0.so ../NLMBase/bin/Release/net5.0
	rm -f NLMGpgpu0.hip.o NLMGpgpu0.so

NLMGpgpu0.so: NLMGpgpu0.hip.o
	hipcc -shared -o NLMGpgpu0.so NLMGpgpu0.hip.o

NLMGpgpu0.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -Wall -Wextra -fpic --stdlib=libstdc++ NLMGpgpu0.hip.cpp

.PHONY: NLMGpgpu0Opti
NLMGpgpu0Opti: NLMGpgpu0Opti.so
	cp NLMGpgpu0Opti.so ../NLMBase
	cp NLMGpgpu0Opti.so ../NLMBase/bin/Release/net5.0
	rm -f NLMGpgpu0Opti.hip.o NLMGpgpu0Opti.so

NLMGpgpu0Opti.so: NLMGpgpu0Opti.hip.o
	hipcc -shared -o NLMGpgpu0Opti.so NLMGpgpu0Opti.hip.o

NLMGpgpu0Opti.hip.o: NLMGpgpu0.hip.cpp NLMGpgpu0.hip.h
	hipcc -c -O3 -funroll-loops -fomit-frame-pointer -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -fpic --stdlib=libstdc++ -o NLMGpgpu0Opti.hip.o NLMGpgpu0.hip.cpp

.PHONY: NLMGpgpu1
NLMGpgpu1: NLMGpgpu1.so
	cp NLMGpgpu1.so ../NLMBase
	cp NLMGpgpu1.so ../NLMBase/bin/Release/net5.0
	rm -f NLMGpgpu1.hip.o NLMGpgpu1.so

NLMGpgpu1.so: NLMGpgpu1.hip.o
	hipcc -shared -o NLMGpgpu1.so NLMGpgpu1.hip.o

NLMGpgpu1.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -Wall -Wextra -fpic --stdlib=libstdc++ NLMGpgpu1.hip.cpp
.PHONY: NLMGpgpu1Opti
NLMGpgpu1Opti: NLMGpgpu1Opti.so
	cp NLMGpgpu1Opti.so ../NLMBase
	cp NLMGpgpu1Opti.so ../NLMBase/bin/Release/net5.0
	rm -f NLMGpgpu1Opti.hip.o NLMGpgpu1Opti.so

NLMGpgpu1Opti.so: NLMGpgpu1Opti.hip.o
	hipcc -shared -o NLMGpgpu1Opti.so NLMGpgpu1Opti.hip.o

NLMGpgpu1Opti.hip.o: NLMGpgpu1.hip.cpp NLMGpgpu1.hip.h
	hipcc -c -O3 -funroll-loops -fomit-frame-pointer -ffast-math -ftree-vectorize -Wall -Wextra -Wno-write-strings -Wno-deprecated -fpic --stdlib=libstdc++ -o NLMGpgpu1Opti.hip.o NLMGpgpu1.hip.cpp