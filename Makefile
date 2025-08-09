HIPCC = hipcc
HIPFLAGS = -I./ --offload-arch=gfx1030 -O2

all: rpc

rpc: rpc.hip
	$(HIPCC) $(HIPFLAGS) $< -o $@

clean:
	rm -f rpc
