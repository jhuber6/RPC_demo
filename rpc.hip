#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#include <shared/rpc.h>

constexpr int PINGS = 100000;

__device__ rpc::Server server;

[[noreturn]] void handle_error(hipError_t err) {
  fprintf(stderr, "HIP error: %s\n", hipGetErrorString(err));
  exit(EXIT_FAILURE);
}

enum rpc_opcode {
  RPC_EXIT,
  RPC_PING,
  RPC_NOOP,
};

__device__ int run = 1;

__global__ void kernel() {
  while (run) {
    auto port = server.try_open(/*num_lanes*/ 1);
    if (!port)
      continue;

    switch (port->get_opcode()) {
    case RPC_EXIT: {
      port->recv([&](rpc::Buffer *buffer, uint32_t) {
        __atomic_store_n(&run, 0, __ATOMIC_RELEASE);
      });
      break;
    }
    case RPC_PING: {
      port->recv_and_send([&](rpc::Buffer *buffer, uint32_t) {
        buffer->data[0] = buffer->data[0] + 1;
      });
      break;
    }
    case RPC_NOOP: {
      port->recv([](rpc::Buffer *, uint32_t) {});
      break;
    }
    }
    port->close();
  }
}

void ping(rpc::Client &client) {
  uint64_t n = 0;
  for (int i = 0; i < PINGS; ++i) {
    rpc::Client::Port port = client.open<RPC_PING>();
    port.send_and_recv(
        [&](rpc::Buffer *buffer, uint32_t) { buffer->data[0] = n; },
        [&](rpc::Buffer *buffer, uint32_t) { n = buffer->data[0]; });
    port.close();
  }
}

void exit(rpc::Client &client) {
  rpc::Client::Port port = client.open<RPC_EXIT>();
  port.send([](rpc::Buffer *buffer, uint32_t) {});
  port.close();
}

int main() {
  void *rpc_server = nullptr;
  if (hipError_t err = hipGetSymbolAddress(&rpc_server, server))
    handle_error(err);

  // Initialize the RPC client and server interface using one lane for the CPU.
  uint32_t num_lanes = 1;
  void *rpc_buffer = nullptr;
  if (hipError_t err = hipHostMalloc(
          &rpc_buffer,
          rpc::Server::allocation_size(num_lanes, 1)))
    handle_error(err);
  rpc::Server server_(1, rpc_buffer);
  rpc::Client client(1, rpc_buffer);

  // Initialize the client on the device so it can communicate with the
  // server.
  if (hipError_t err = hipMemcpy(rpc_server, &server_, sizeof(rpc::Server),
                                 hipMemcpyHostToDevice))
    handle_error(err);

  hipStream_t stream;
  if (hipError_t err = hipStreamCreate(&stream))
    handle_error(err);

  // Execute the kernel.
  hipEvent_t start, stop;
  if (hipError_t err = hipEventCreate(&start))
    handle_error(err);
  if (hipError_t err = hipEventCreate(&stop))
    handle_error(err);

  if (hipError_t err = hipEventRecord(start, stream))
    handle_error(err);
  hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, stream);

  // While the kernel is executing, submit work for the RPC server to do.
  // Requires non-blocking HIP kernels but avoids a separate thread.
  ping(client);
  exit(client);

  if (hipError_t err = hipEventRecord(stop, stream))
    handle_error(err);
  if (hipError_t err = hipEventSynchronize(stop))
    handle_error(err);

  float elapsed;
  if (hipError_t err = hipEventElapsedTime(&elapsed, start, stop))
    handle_error(err);
  printf("Took %f ms to ping %d times\n", elapsed, PINGS);
}
