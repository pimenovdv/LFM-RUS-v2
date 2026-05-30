# Configuration Files

The pipeline stages are configured using YAML files located in the `configs/` directory.

### Tokenizer configuration (`tokenizer.yaml`)
Configures the tokenizer training parameters, vocabulary size, lexical initialization methods (e.g., FOCUS), and paths and ratios of datasets used for training.

### Pruning configuration (`pruning.yaml`)
Specifies the model to prune, the calibration dataset path, the minimum frequency of tokens to keep, and the output directory for the pruned model.

### Data Prep configuration (`data_prep.yaml`)
Contains settings for the data cleaning and deduplication pipeline. Defines input and output paths, MinHash LSH configuration (n-grams, buckets, etc.), and settings for various filters (spam, SEO, cyclic).

### Continual Pre-Training (`cpt.yaml`)
Defines the hyper-parameters for the CPT stage, such as batch sizes, learning rates, gradient accumulation, sequence length, and checkpointing intervals.

### Supervised Fine-Tuning (`sft.yaml`)
Provides configuration for SFT, including max sequence lengths, sequence packing, packing arguments, and model saving locations.

### Alignment configuration (`alignment.yaml`)
Configures alignment methods (DPO or GRPO). Allows selecting the reward functions to use (like accuracy or variance) and general training parameters.

### Task SFT configuration (`task_sft.yaml`)
Includes configurations for the final targeted fine-tuning, including the specific target modules to use if PEFT/LoRA is enabled.
