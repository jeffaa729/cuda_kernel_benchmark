BUILD_DIR ?= build
CMAKE ?= cmake

.PHONY: all configure run test clean

all: configure
	$(CMAKE) --build $(BUILD_DIR) --parallel

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

run: all
	./$(BUILD_DIR)/bin/vector_add_benchmark

test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
