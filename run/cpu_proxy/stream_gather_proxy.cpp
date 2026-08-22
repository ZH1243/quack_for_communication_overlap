#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}

void check_driver(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (message == nullptr ? "unknown CUDA driver error" : message));
}

struct Options {
  int device = 0;
  std::string ipc_handle;
  std::size_t table_offset_bytes = 0;
  std::size_t ready_offset_bytes = 0;
  int table_rows = 0;
  int table_width = 0;
  int experts = 0;
  int routes_per_buffer = 0;
  int num_input_buffers = 0;
  int output_dim = 0;
  int tile_m = 0;
  int tile_n = 0;
  int cluster_m = 0;
  int max_swizzle_size = 0;
  int entries_per_flush = 0;
  int interval_us = 0;
  std::size_t dma_kick_bytes = 0;
  bool balanced = false;
  bool round_robin = false;
  std::string flag_mode;
};

void validate_options(const Options& options) {
  if (options.table_rows <= 0 || options.table_width <= 0 || options.experts <= 0 ||
      options.routes_per_buffer <= 0 || options.num_input_buffers < 2 ||
      options.output_dim <= 0 || options.tile_m <= 0 || options.tile_n <= 0 ||
      options.cluster_m <= 0 || options.max_swizzle_size <= 0 ||
      options.entries_per_flush <= 0 || options.interval_us < 0) {
    throw std::runtime_error("table and flush dimensions must be positive (interval may be zero)");
  }
  if (options.table_width != 2 + 2 * options.num_input_buffers) {
    throw std::runtime_error("table width does not match the multi-buffer row format");
  }
  if (options.flag_mode != "memcpy" && options.flag_mode != "stream-write") {
    throw std::runtime_error("--flag-mode must be memcpy or stream-write");
  }
}

std::unordered_map<std::string, std::string> parse_pairs(int argc, char** argv) {
  std::unordered_map<std::string, std::string> values;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc || std::string(argv[i]).rfind("--", 0) != 0) {
      throw std::runtime_error("arguments must be supplied as --name value pairs");
    }
    values.emplace(std::string(argv[i]).substr(2), argv[i + 1]);
  }
  return values;
}

const std::string& required(const std::unordered_map<std::string, std::string>& values,
                            const std::string& name) {
  const auto it = values.find(name);
  if (it == values.end()) {
    throw std::runtime_error("missing required argument --" + name);
  }
  return it->second;
}

template <typename T>
T integer_option(const std::unordered_map<std::string, std::string>& values,
                 const std::string& name) {
  const auto text = required(values, name);
  std::size_t consumed = 0;
  const auto parsed = std::stoll(text, &consumed);
  bool out_of_range = false;
  if constexpr (std::is_unsigned_v<T>) {
    out_of_range =
        parsed < 0 || static_cast<unsigned long long>(parsed) > std::numeric_limits<T>::max();
  } else {
    out_of_range = parsed < std::numeric_limits<T>::min() ||
                   parsed > std::numeric_limits<T>::max();
  }
  if (consumed != text.size() || out_of_range) {
    throw std::runtime_error("invalid integer for --" + name + ": " + text);
  }
  return static_cast<T>(parsed);
}

Options parse_options(int argc, char** argv) {
  const auto values = parse_pairs(argc, argv);
  Options options;
  options.device = integer_option<int>(values, "device");
  options.ipc_handle = required(values, "ipc-handle");
  options.table_offset_bytes = integer_option<std::size_t>(values, "table-offset-bytes");
  options.ready_offset_bytes = integer_option<std::size_t>(values, "ready-offset-bytes");
  options.table_rows = integer_option<int>(values, "table-rows");
  options.table_width = integer_option<int>(values, "table-width");
  options.experts = integer_option<int>(values, "experts");
  options.routes_per_buffer = integer_option<int>(values, "routes-per-buffer");
  options.num_input_buffers = integer_option<int>(values, "num-input-buffers");
  options.output_dim = integer_option<int>(values, "output-dim");
  options.tile_m = integer_option<int>(values, "tile-m");
  options.tile_n = integer_option<int>(values, "tile-n");
  options.cluster_m = integer_option<int>(values, "cluster-m");
  options.max_swizzle_size = integer_option<int>(values, "max-swizzle-size");
  options.entries_per_flush = integer_option<int>(values, "entries-per-flush");
  options.interval_us = integer_option<int>(values, "interval-us");
  options.dma_kick_bytes = integer_option<std::size_t>(values, "dma-kick-bytes");
  options.balanced = integer_option<int>(values, "balanced") != 0;
  options.round_robin = integer_option<int>(values, "round-robin") != 0;
  options.flag_mode = required(values, "flag-mode");

  validate_options(options);
  return options;
}

cudaIpcMemHandle_t decode_ipc_handle(const std::string& text) {
  if (text.size() % 2 != 0) {
    throw std::runtime_error(
        "CUDA IPC handle must contain an even number of hexadecimal characters");
  }

  std::vector<unsigned char> encoded(text.size() / 2);
  for (std::size_t i = 0; i < encoded.size(); ++i) {
    const auto pair = text.substr(2 * i, 2);
    std::size_t consumed = 0;
    unsigned long value = 0;
    try {
      value = std::stoul(pair, &consumed, 16);
    } catch (const std::exception&) {
      throw std::runtime_error("invalid hexadecimal CUDA IPC handle");
    }
    if (consumed != 2) {
      throw std::runtime_error("invalid hexadecimal CUDA IPC handle");
    }
    encoded[i] = static_cast<unsigned char>(value);
  }

  constexpr std::size_t raw_bytes = sizeof(cudaIpcMemHandle_t);
  std::size_t raw_offset = 0;
  if (encoded.size() == raw_bytes) {
    // Legacy PyTorch exported cudaIpcMemHandle_t without an envelope.
  } else if (encoded.size() == raw_bytes + 2 &&
             encoded[1] == static_cast<unsigned char>('c')) {
    // Recent PyTorch versions prefix allocator IPC payloads with version and type.
    raw_offset = 2;
  } else if (encoded.size() >= 2 && encoded[1] == static_cast<unsigned char>('e')) {
    throw std::runtime_error(
        "PyTorch expandable-segment IPC handles cannot be opened with cudaIpcOpenMemHandle; "
        "disable expandable segments for this process");
  } else {
    throw std::runtime_error(
        "unsupported CUDA IPC handle encoding: expected 64 raw bytes or a 66-byte "
        "versioned cudaMalloc handle");
  }

  cudaIpcMemHandle_t handle{};
  auto* output = reinterpret_cast<unsigned char*>(&handle);
  std::copy_n(encoded.begin() + raw_offset, raw_bytes, output);
  return handle;
}

using Ranges = std::vector<std::pair<std::int32_t, std::int32_t>>;

std::vector<int> sequential_allocations(const std::vector<int>& remaining, int capacity) {
  std::vector<int> allocations(remaining.size(), 0);
  for (std::size_t j = 0; j < remaining.size() && capacity != 0; ++j) {
    allocations[j] = std::min(remaining[j], capacity);
    capacity -= allocations[j];
  }
  return allocations;
}

std::vector<int> balanced_allocations(const std::vector<int>& remaining, int capacity) {
  std::vector<int> allocations(remaining.size(), 0);
  auto available = remaining;
  std::vector<int> active;
  for (std::size_t j = 0; j < available.size(); ++j) {
    if (available[j] != 0) {
      active.push_back(static_cast<int>(j));
    }
  }
  while (capacity != 0 && !active.empty()) {
    const int share = capacity / static_cast<int>(active.size());
    int minimum = std::numeric_limits<int>::max();
    for (const int j : active) {
      minimum = std::min(minimum, available[j]);
    }
    if (minimum <= share) {
      for (const int j : active) {
        allocations[j] += minimum;
        available[j] -= minimum;
      }
      capacity -= minimum * static_cast<int>(active.size());
      active.erase(std::remove_if(active.begin(), active.end(), [&](int j) {
                     return available[j] == 0;
                   }),
                   active.end());
      continue;
    }

    for (const int j : active) {
      allocations[j] += share;
      available[j] -= share;
    }
    capacity -= share * static_cast<int>(active.size());
    if (capacity != 0) {
      auto owner = std::find_if(active.begin(), active.end(),
                                [&](int j) { return available[j] >= capacity; });
      if (owner != active.end()) {
        allocations[*owner] += capacity;
        capacity = 0;
      } else {
        for (const int j : active) {
          const int allocation = std::min(available[j], capacity);
          allocations[j] += allocation;
          capacity -= allocation;
          if (capacity == 0) {
            break;
          }
        }
      }
    }
  }
  return allocations;
}

struct PinnedTable {
  std::int32_t* data = nullptr;
  int rows = 0;
  int width = 0;
};

PinnedTable build_table(const Options& options) {
  const int base = options.routes_per_buffer / options.experts;
  const int remainder = options.routes_per_buffer % options.experts;
  std::vector<std::vector<int>> counts(options.num_input_buffers,
                                       std::vector<int>(options.experts));
  std::vector<std::vector<int>> offsets(options.num_input_buffers,
                                        std::vector<int>(options.experts + 1, 0));
  for (int buffer = 0; buffer < options.num_input_buffers; ++buffer) {
    for (int expert = 0; expert < options.experts; ++expert) {
      const int rotated =
          (expert - (buffer % options.experts) + options.experts) % options.experts;
      counts[buffer][expert] = base + static_cast<int>(rotated < remainder);
      offsets[buffer][expert + 1] = offsets[buffer][expert] + counts[buffer][expert];
    }
  }

  const int cluster_rows = options.tile_m * options.cluster_m;
  std::vector<std::vector<Ranges>> cluster_ranges(options.experts);
  for (int expert = 0; expert < options.experts; ++expert) {
    std::vector<int> consumed(options.num_input_buffers, 0);
    int total = 0;
    for (int buffer = 0; buffer < options.num_input_buffers; ++buffer) {
      total += counts[buffer][expert];
    }
    int consumed_total = 0;
    while (consumed_total < total) {
      std::vector<int> remaining(options.num_input_buffers);
      for (int buffer = 0; buffer < options.num_input_buffers; ++buffer) {
        remaining[buffer] = counts[buffer][expert] - consumed[buffer];
      }
      const int capacity = std::min(cluster_rows, total - consumed_total);
      const auto allocations = options.balanced
                                   ? balanced_allocations(remaining, capacity)
                                   : sequential_allocations(remaining, capacity);
      Ranges ranges;
      ranges.reserve(options.num_input_buffers);
      for (int buffer = 0; buffer < options.num_input_buffers; ++buffer) {
        const int start = offsets[buffer][expert] + consumed[buffer];
        const int end = start + allocations[buffer];
        ranges.emplace_back(start, end);
        consumed[buffer] += allocations[buffer];
        consumed_total += allocations[buffer];
      }
      cluster_ranges[expert].push_back(std::move(ranges));
    }
  }

  const int clusters_n = (options.output_dim + options.tile_n - 1) / options.tile_n;
  const int group_size = std::min(options.max_swizzle_size, clusters_n);
  if (clusters_n % group_size != 0) {
    throw std::runtime_error("clusters_n must be divisible by the work-table group size");
  }
  const int num_n_groups = clusters_n / group_size;
  std::size_t row_count = 0;
  for (const auto& expert_ranges : cluster_ranges) {
    row_count += expert_ranges.size() * static_cast<std::size_t>(num_n_groups);
  }
  if (row_count != static_cast<std::size_t>(options.table_rows)) {
    throw std::runtime_error("proxy computed " + std::to_string(row_count) +
                             " table rows, Python allocated " +
                             std::to_string(options.table_rows));
  }

  PinnedTable table;
  table.rows = options.table_rows;
  table.width = options.table_width;
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&table.data),
                           row_count * table.width * sizeof(std::int32_t),
                           cudaHostAllocDefault),
             "cudaHostAlloc(table)");

  std::size_t cursor = 0;
  auto emit = [&](int expert, int n_group, const Ranges& ranges) {
    auto* row = table.data + cursor * table.width;
    row[0] = expert;
    row[1] = n_group * group_size;
    for (int buffer = 0; buffer < options.num_input_buffers; ++buffer) {
      row[2 + 2 * buffer] = ranges[buffer].first;
      row[3 + 2 * buffer] = ranges[buffer].second;
    }
    ++cursor;
  };

  if (options.round_robin) {
    std::size_t max_clusters = 0;
    for (const auto& expert_ranges : cluster_ranges) {
      max_clusters = std::max(max_clusters, expert_ranges.size());
    }
    for (std::size_t cluster = 0; cluster < max_clusters; ++cluster) {
      for (int expert = 0; expert < options.experts; ++expert) {
        if (cluster >= cluster_ranges[expert].size()) {
          continue;
        }
        for (int n_group = 0; n_group < num_n_groups; ++n_group) {
          emit(expert, n_group, cluster_ranges[expert][cluster]);
        }
      }
    }
  } else {
    for (int expert = 0; expert < options.experts; ++expert) {
      for (int n_group = 0; n_group < num_n_groups; ++n_group) {
        if ((n_group & 1) == 0) {
          for (const auto& ranges : cluster_ranges[expert]) {
            emit(expert, n_group, ranges);
          }
        } else {
          for (auto it = cluster_ranges[expert].rbegin();
               it != cluster_ranges[expert].rend(); ++it) {
            emit(expert, n_group, *it);
          }
        }
      }
    }
  }
  if (cursor != row_count) {
    throw std::runtime_error("internal table construction row-count mismatch");
  }
  return table;
}

class FlushProxy {
 public:
  FlushProxy(Options options, std::int32_t* device_table, std::int32_t* device_ready)
      : options_(std::move(options)), device_table_(device_table), device_ready_(device_ready) {}

  FlushProxy(const FlushProxy&) = delete;
  FlushProxy& operator=(const FlushProxy&) = delete;

  ~FlushProxy() {
    if (stream_ != nullptr) {
      cudaStreamSynchronize(stream_);
    }
    if (ready_values_ != nullptr) {
      cudaFreeHost(ready_values_);
    }
    if (kick_device_ != nullptr) {
      cudaFree(kick_device_);
    }
    if (kick_host_ != nullptr) {
      cudaFreeHost(kick_host_);
    }
    if (table_.data != nullptr) {
      cudaFreeHost(table_.data);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  void initialize() {
    table_ = build_table(options_);
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreate");
    batches_ = (table_.rows + options_.entries_per_flush - 1) / options_.entries_per_flush;
    if (options_.flag_mode == "memcpy") {
      check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&ready_values_),
                               batches_ * sizeof(std::int32_t), cudaHostAllocDefault),
                 "cudaHostAlloc(readiness values)");
      for (int batch = 0; batch < batches_; ++batch) {
        ready_values_[batch] =
            std::min((batch + 1) * options_.entries_per_flush, table_.rows);
      }
    }
    if (options_.dma_kick_bytes != 0) {
      check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&kick_host_), options_.dma_kick_bytes,
                               cudaHostAllocDefault),
                 "cudaHostAlloc(DMA kick source)");
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&kick_device_), options_.dma_kick_bytes),
                 "cudaMalloc(DMA kick destination)");
      std::fill_n(kick_host_, options_.dma_kick_bytes, 0);
    }
  }

  void run_once() {
    auto deadline = std::chrono::steady_clock::now();
    if (options_.dma_kick_bytes != 0) {
      // CUDA uses a front-end inline path for sufficiently small HtoD copies.
      // In a separate CUDA context that path can wait milliseconds behind a
      // resident persistent kernel. A preceding copy-engine-sized transfer
      // activates this stream without publishing table rows early.
      check_cuda(cudaMemcpyAsync(kick_device_, kick_host_, options_.dma_kick_bytes,
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync(DMA kick)");
    }
    for (int batch = 0; batch < batches_; ++batch) {
      if (batch != 0 && options_.interval_us != 0) {
        deadline += std::chrono::microseconds(options_.interval_us);
        std::this_thread::sleep_until(deadline);
      }
      const int begin = batch * options_.entries_per_flush;
      const int end = std::min(begin + options_.entries_per_flush, table_.rows);
      const std::size_t entries = static_cast<std::size_t>(end - begin) * table_.width;
      check_cuda(cudaMemcpyAsync(device_table_ + static_cast<std::size_t>(begin) * table_.width,
                                 table_.data + static_cast<std::size_t>(begin) * table_.width,
                                 entries * sizeof(std::int32_t), cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync(table rows)");
      if (options_.flag_mode == "memcpy") {
        check_cuda(cudaMemcpyAsync(device_ready_, ready_values_ + batch, sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream_),
                   "cudaMemcpyAsync(readiness flag)");
      } else {
        check_driver(cuStreamWriteValue32(
                         reinterpret_cast<CUstream>(stream_),
                         static_cast<CUdeviceptr>(
                             reinterpret_cast<std::uintptr_t>(device_ready_)),
                         static_cast<cuuint32_t>(end), CU_STREAM_WRITE_VALUE_DEFAULT),
                     "cuStreamWriteValue32(readiness flag)");
      }
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
  }

  int rows() const { return table_.rows; }
  int width() const { return table_.width; }

 private:
  Options options_;
  PinnedTable table_;
  std::int32_t* device_table_ = nullptr;
  std::int32_t* device_ready_ = nullptr;
  cudaStream_t stream_ = nullptr;
  int batches_ = 0;
  std::int32_t* ready_values_ = nullptr;
  unsigned char* kick_host_ = nullptr;
  unsigned char* kick_device_ = nullptr;
};

void initialize_cuda(int device) {
  check_driver(cuInit(0), "cuInit");
  check_cuda(cudaSetDevice(device), "cudaSetDevice");
  check_cuda(cudaFree(nullptr), "initialize CUDA runtime context");
}

void run(const Options& options) {
  initialize_cuda(options.device);

  const auto handle = decode_ipc_handle(options.ipc_handle);
  void* allocation = nullptr;
  check_cuda(cudaIpcOpenMemHandle(&allocation, handle, cudaIpcMemLazyEnablePeerAccess),
             "cudaIpcOpenMemHandle");
  auto* allocation_bytes = static_cast<unsigned char*>(allocation);
  auto* device_table = reinterpret_cast<std::int32_t*>(
      allocation_bytes + options.table_offset_bytes);
  auto* device_ready = reinterpret_cast<std::int32_t*>(
      allocation_bytes + options.ready_offset_bytes);

  try {
    FlushProxy proxy(options, device_table, device_ready);
    proxy.initialize();
    std::cout << "READY rows=" << proxy.rows() << " width=" << proxy.width()
              << " dma-kick-bytes=" << options.dma_kick_bytes << std::endl;
    std::string command;
    while (std::getline(std::cin, command)) {
      if (command == "QUIT") {
        break;
      }
      if (command != "GO") {
        throw std::runtime_error("expected GO or QUIT on stdin");
      }
      proxy.run_once();
      std::cout << "DONE rows=" << proxy.rows() << std::endl;
    }
  } catch (...) {
    cudaIpcCloseMemHandle(allocation);
    throw;
  }
  check_cuda(cudaIpcCloseMemHandle(allocation), "cudaIpcCloseMemHandle");
}

struct ThreadProxyState {
  int device = 0;
  std::unique_ptr<FlushProxy> proxy;
  std::mutex mutex;
  std::condition_variable condition;
  bool requested = false;
  bool outstanding = false;
  bool completed = false;
  bool stopping = false;
  std::string worker_error;
  std::thread worker;

  ThreadProxyState(int device_value, std::unique_ptr<FlushProxy> proxy_value)
      : device(device_value), proxy(std::move(proxy_value)),
        worker(&ThreadProxyState::worker_loop, this) {}

  ThreadProxyState(const ThreadProxyState&) = delete;
  ThreadProxyState& operator=(const ThreadProxyState&) = delete;

  ~ThreadProxyState() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
    }
    condition.notify_one();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void start() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (outstanding) {
        throw std::runtime_error("thread proxy already has a flush in flight");
      }
      worker_error.clear();
      requested = true;
      outstanding = true;
      completed = false;
    }
    condition.notify_one();
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mutex);
    if (!outstanding) {
      throw std::runtime_error("thread proxy has no flush in flight");
    }
    condition.wait(lock, [this] { return completed; });
    outstanding = false;
    if (!worker_error.empty()) {
      throw std::runtime_error(worker_error);
    }
  }

  void worker_loop() noexcept {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      condition.wait(lock, [this] { return requested || stopping; });
      if (stopping && !requested) {
        return;
      }
      requested = false;
      lock.unlock();

      std::string error;
      try {
        check_cuda(cudaSetDevice(device), "cudaSetDevice(worker thread)");
        proxy->run_once();
      } catch (const std::exception& exception) {
        error = exception.what();
      } catch (...) {
        error = "unknown exception in thread proxy worker";
      }

      lock.lock();
      worker_error = std::move(error);
      completed = true;
      lock.unlock();
      condition.notify_all();
    }
  }
};

void write_error(char* error, std::size_t capacity, const std::string& message) {
  if (error == nullptr || capacity == 0) {
    return;
  }
  const auto length = std::min(capacity - 1, message.size());
  std::memcpy(error, message.data(), length);
  error[length] = '\0';
}

Options make_thread_options(int device, int table_rows, int table_width, int experts,
                            int routes_per_buffer, int num_input_buffers, int output_dim,
                            int tile_m, int tile_n, int cluster_m, int max_swizzle_size,
                            int entries_per_flush, int interval_us,
                            std::size_t dma_kick_bytes, int balanced, int round_robin,
                            int flag_mode) {
  Options options;
  options.device = device;
  options.table_rows = table_rows;
  options.table_width = table_width;
  options.experts = experts;
  options.routes_per_buffer = routes_per_buffer;
  options.num_input_buffers = num_input_buffers;
  options.output_dim = output_dim;
  options.tile_m = tile_m;
  options.tile_n = tile_n;
  options.cluster_m = cluster_m;
  options.max_swizzle_size = max_swizzle_size;
  options.entries_per_flush = entries_per_flush;
  options.interval_us = interval_us;
  options.dma_kick_bytes = dma_kick_bytes;
  options.balanced = balanced != 0;
  options.round_robin = round_robin != 0;
  if (flag_mode == 0) {
    options.flag_mode = "memcpy";
  } else if (flag_mode == 1) {
    options.flag_mode = "stream-write";
  } else {
    throw std::runtime_error("thread proxy flag mode must be 0 or 1");
  }
  validate_options(options);
  return options;
}

}  // namespace

#ifdef QUACK_STREAM_GATHER_PROXY_LIBRARY
extern "C" void* stream_gather_thread_proxy_create(
    int device, std::uintptr_t device_table, std::uintptr_t device_ready, int table_rows,
    int table_width, int experts, int routes_per_buffer, int num_input_buffers, int output_dim,
    int tile_m, int tile_n, int cluster_m, int max_swizzle_size, int entries_per_flush,
    int interval_us, std::size_t dma_kick_bytes, int balanced, int round_robin, int flag_mode,
    char* error, std::size_t error_capacity) {
  try {
    if (device_table == 0 || device_ready == 0) {
      throw std::runtime_error("thread proxy device pointers must be non-null");
    }
    auto options = make_thread_options(
        device, table_rows, table_width, experts, routes_per_buffer, num_input_buffers,
        output_dim, tile_m, tile_n, cluster_m, max_swizzle_size, entries_per_flush, interval_us,
        dma_kick_bytes, balanced, round_robin, flag_mode);
    initialize_cuda(device);
    auto proxy = std::make_unique<FlushProxy>(
        options, reinterpret_cast<std::int32_t*>(device_table),
        reinterpret_cast<std::int32_t*>(device_ready));
    proxy->initialize();
    auto state = std::make_unique<ThreadProxyState>(device, std::move(proxy));
    return state.release();
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return nullptr;
  }
}

extern "C" int stream_gather_thread_proxy_start(void* handle, char* error,
                                                  std::size_t error_capacity) {
  try {
    if (handle == nullptr) {
      throw std::runtime_error("thread proxy handle is null");
    }
    auto* state = static_cast<ThreadProxyState*>(handle);
    state->start();
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" int stream_gather_thread_proxy_wait(void* handle, char* error,
                                                 std::size_t error_capacity) {
  try {
    if (handle == nullptr) {
      throw std::runtime_error("thread proxy handle is null");
    }
    auto* state = static_cast<ThreadProxyState*>(handle);
    state->wait();
    return 0;
  } catch (const std::exception& exception) {
    write_error(error, error_capacity, exception.what());
    return 1;
  }
}

extern "C" void stream_gather_thread_proxy_destroy(void* handle) {
  delete static_cast<ThreadProxyState*>(handle);
}
#else
int main(int argc, char** argv) {
  try {
    run(parse_options(argc, argv));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "stream_gather_proxy: " << error.what() << '\n';
    return 1;
  }
}
#endif
