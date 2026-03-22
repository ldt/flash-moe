/*
 * model_config.h — Model configuration selector
 *
 * This header routes to the correct model-specific config based on the
 * MODEL_ID preprocessor define, set via the Makefile's MODEL= variable.
 *
 * Usage:
 *   make MODEL=qwen3.5-397b    # Build for Qwen3.5-397B-A17B (default)
 *   make MODEL=qwen3.5-122b    # Build for Qwen3.5-122B-A10B
 *
 * Or copy a config directly:
 *   cp configs/qwen3.5-122b.h model_config_active.h && make
 *
 * Available configs:
 *   configs/qwen3.5-397b.h   — 397B params, 60 layers, 4096 hidden, 512 experts
 *   configs/qwen3.5-122b.h   — 122B params, 48 layers, 3072 hidden, 256 experts
 *
 * To add a new model, create configs/your-model.h following the same pattern
 * and add a MODEL_ID case below.
 */

#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

#if defined(MODEL_QWEN3_5_122B)
  #include "configs/qwen3.5-122b.h"
#elif defined(MODEL_QWEN3_5_397B) || !defined(MODEL_QWEN3_5_122B)
  /* Default: Qwen3.5-397B-A17B */
  #include "configs/qwen3.5-397b.h"
#endif

#endif // MODEL_CONFIG_H
