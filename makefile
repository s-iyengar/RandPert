CPP = /opt/homebrew/opt/llvm/bin/clang++
CPPFLAGS = -I/opt/homebrew/opt/llvm/include -fopenmp
LDFLAGS=  -L/opt/homebrew/opt/llvm/lib  -rpath /opt/homebrew/opt/llvm/lib/c++ -lomp
SETTINGS = -std=c++17 -O3

Negative1YHillFunction: Negative1YHillFunction.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) Negative1YHillFunction.cpp -o fsnfb1 $(LDFLAGS) -v

NegYHF: NegYHF_gen.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) NegYHF_gen.cpp -o nyhf $(LDFLAGS) -v

bothfb: BothFBSims.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) BothFBSims.cpp -o bfb $(LDFLAGS) -v

weirdfb: weirdfb.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) weirdfb.cpp -o wfb $(LDFLAGS) -v

andgate_pdist: AndGate_pdists.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) AndGate_pdists.cpp -o andp $(LDFLAGS) -v

hardcoding: bothfb_hardcodedsystem.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) bothfb_hardcodedsystem.cpp -o bfb $(LDFLAGS) -v

inthill: IntermediateHillFunction.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) IntermediateHillFunction.cpp -o inthill $(LDFLAGS) -v

directy_pdist: Intermediate_x_not_y.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) Intermediate_x_not_y.cpp -o dypdist $(LDFLAGS) -v

directx_pdist: Intermediate_directx_indirecty.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) Intermediate_directx_indirecty.cpp -o dxpdist $(LDFLAGS) -v

inthill_withpdist: IntermediateHillFunction_withdist.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) IntermediateHillFunction_withdist.cpp -o inthill_pdist $(LDFLAGS) -v

FiniteSample: FiniteSampleTests.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) FiniteSampleTests.cpp -o finitesamp $(LDFLAGS) -v

orgate_pdist: Orgate_pdists.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) Orgate_pdists.cpp -o ogpd $(LDFLAGS) -v


manypoints: IntermediateHillFunction_manypoints.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) IntermediateHillFunction_manypoints.cpp -o inthill $(LDFLAGS) -v

viol_pertscale: IntermediateHillFunction_withdist_pertscales.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) IntermediateHillFunction_withdist_pertscales.cpp -o vpp $(LDFLAGS) -v

changeK: changing_K.cpp
	$(CPP) $(SETTINGS) $(CPPFLAGS) changing_K.cpp -o ck $(LDFLAGS) -v

clean:
	rm fsnfb1
	rm nyhf
	rm bfb