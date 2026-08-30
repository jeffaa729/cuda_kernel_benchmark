BUILD_DIR ?= build
CMAKE ?= cmake
BENCH ?= all
ARGS ?=

.PHONY: all configure run vector_add transpose reduction gemm softmax conv2d test clean rmsnorm swiglu rope embedding adamw global_norm cross_entropy causal_softmax attention

all: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

run: all
	./$(BUILD_DIR)/bin/cuda_benchmarks $(BENCH) $(ARGS)

vector_add: all
	./$(BUILD_DIR)/bin/cuda_benchmarks vector_add $(ARGS)

transpose: all
	./$(BUILD_DIR)/bin/cuda_benchmarks transpose $(ARGS)

reduction: all
	./$(BUILD_DIR)/bin/cuda_benchmarks reduction $(ARGS)

gemm: all
	./$(BUILD_DIR)/bin/cuda_benchmarks gemm $(ARGS)

softmax: all
	./$(BUILD_DIR)/bin/cuda_benchmarks softmax $(ARGS)

conv2d: all
	./$(BUILD_DIR)/bin/cuda_benchmarks conv2d $(ARGS)

rmsnorm: all
	./$(BUILD_DIR)/bin/cuda_benchmarks rmsnorm $(ARGS)

swiglu: all
	./$(BUILD_DIR)/bin/cuda_benchmarks swiglu $(ARGS)

rope: all
	./$(BUILD_DIR)/bin/cuda_benchmarks rope $(ARGS)

embedding: all
	./$(BUILD_DIR)/bin/cuda_benchmarks embedding $(ARGS)

adamw: all
	./$(BUILD_DIR)/bin/cuda_benchmarks adamw $(ARGS)

global_norm: all
	./$(BUILD_DIR)/bin/cuda_benchmarks global_norm $(ARGS)

cross_entropy: all
	./$(BUILD_DIR)/bin/cuda_benchmarks cross_entropy $(ARGS)

causal_softmax: all
	./$(BUILD_DIR)/bin/cuda_benchmarks causal_softmax $(ARGS)

attention: all
	./$(BUILD_DIR)/bin/cuda_benchmarks attention $(ARGS)

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
