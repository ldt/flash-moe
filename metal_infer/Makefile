# Makefile for metal_infer — Pure C/Metal MoE inference engine
#
# Model selection (configs/ directory):
#   make MODEL=qwen3.5-397b   — Qwen3.5-397B-A17B (default)
#   make MODEL=qwen3.5-122b   — Qwen3.5-122B-A10B
#
# Targets:
#   make          — build metal_infer binary
#   make run      — single expert forward pass
#   make verify   — verify Metal vs CPU reference
#   make bench    — benchmark single expert (10 iterations)
#   make moe      — full MoE forward pass (K experts, single layer)
#   make moebench — benchmark MoE (10 iterations)
#   make full     — full N-layer MoE forward pass (K=4)
#   make fullbench — benchmark full N-layer forward (3 iterations)
#   make chat     — build interactive chat TUI
#   make clean    — remove build artifacts
#
# Note: Metal shaders are compiled from source at runtime via
# MTLDevice newLibraryWithSource:, so no offline metal compiler needed.

CC = clang
CFLAGS = -O2 -Wall -Wextra -fobjc-arc -DACCELERATE_NEW_LAPACK
FRAMEWORKS = -framework Metal -framework Foundation -framework Accelerate
LDFLAGS = -lpthread -lcompression

# Model selection: make MODEL=qwen3.5-397b (default) or make MODEL=qwen3.5-122b
MODEL ?= qwen3.5-397b
ifeq ($(MODEL),qwen3.5-122b)
  MODEL_FLAG = -DMODEL_QWEN3_5_122B
else
  MODEL_FLAG = -DMODEL_QWEN3_5_397B
endif

TARGET = metal_infer
MAIN_SRC = main.m

# Optional: offline shader compilation (faster startup, but not required)
METALC = xcrun -sdk macosx metal
METALLIB_TOOL = xcrun -sdk macosx metallib
SHADER_SRC = shaders.metal
SHADER_AIR = shaders.air
SHADER_LIB = shaders.metallib

# Inference engine (complete forward pass)
INFER_TARGET = infer
INFER_SRC = infer.m

# Chat TUI (interactive multi-turn)
CHAT_TARGET = chat
CHAT_SRC = chat.m

.PHONY: all clean run verify bench moe moebench full fullbench fast metallib infer infer-run chat-run build-chat

all: $(TARGET) $(INFER_TARGET)

# Build the binary (shaders compiled at runtime from source)
$(TARGET): $(MAIN_SRC) $(SHADER_SRC)
	$(CC) $(CFLAGS) $(FRAMEWORKS) $(LDFLAGS) $(MAIN_SRC) -o $(TARGET)

# Optional: pre-compile shaders (not required — runtime compilation is the default)
metallib: $(SHADER_LIB)

$(SHADER_AIR): $(SHADER_SRC)
	$(METALC) -c $(SHADER_SRC) -o $(SHADER_AIR)

$(SHADER_LIB): $(SHADER_AIR)
	$(METALLIB_TOOL) $(SHADER_AIR) -o $(SHADER_LIB)

# Build the inference engine (model selected via MODEL= variable)
$(INFER_TARGET): $(INFER_SRC) model_config.h configs/$(MODEL).h
	$(CC) $(CFLAGS) $(MODEL_FLAG) $(FRAMEWORKS) $(LDFLAGS) $(INFER_SRC) -o $(INFER_TARGET)

# Build the chat client (thin HTTP/SSE client + linenoise line editor)
$(CHAT_TARGET): $(CHAT_SRC) linenoise.c linenoise.h
	$(CC) -O2 -Wall -fobjc-arc -framework Foundation $(CHAT_SRC) linenoise.c -o $(CHAT_TARGET)

clean:
	rm -f $(TARGET) $(INFER_TARGET) $(CHAT_TARGET) $(SHADER_AIR) $(SHADER_LIB)

# Run targets
run: $(TARGET)
	./$(TARGET) --layer 0 --expert 0

verify: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --verify

fast: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --verify

bench: $(TARGET)
	./$(TARGET) --layer 0 --expert 0 --fast --benchmark

moe: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe

moebench: $(TARGET)
	./$(TARGET) --layer 0 --fast --moe --benchmark

full: $(TARGET)
	./$(TARGET) --fast --full --k 4

fullbench: $(TARGET)
	./$(TARGET) --fast --full --k 4 --benchmark

# Inference engine targets
build-infer: $(INFER_TARGET)

infer-run: $(INFER_TARGET)
	./$(INFER_TARGET) --prompt "Hello, what is" --tokens 20 --k 4

# Chat TUI targets (use: make chat)

chat-run: $(CHAT_TARGET)
	./$(CHAT_TARGET) --k 4
