# invoke SourceDir generated makefile for hello.p64P
hello.p64P: .libraries,hello.p64P
.libraries,hello.p64P: package/cfg/hello_p64P.xdl
	$(MAKE) -f N:\workspace-ccs\basico_main/src/makefile.libs

clean::
	$(MAKE) -f N:\workspace-ccs\basico_main/src/makefile.libs clean

