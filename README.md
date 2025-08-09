# HIP demo for RPC server run on the GPU

Simple example of running a server on the GPU using the RPC interface. The
`<gpuintrin.h>` headers are supposed to be provided by the compiler, but are
provided here for older compilers. Additionally newer clang correctly handles
the `__has_builtin` calls, but I fixed that in source again for older compilers.

## Building

Simple makefile runs a ping test.

```console
$ make
$ ./rpc
```
