BUILD_DIR ?= build
CMAKE ?= cmake
BENCH ?= all
ARGS ?=

.PHONY: all configure run vector_add transpose test clean

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

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
