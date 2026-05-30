import os
import shutil
from transformers import AutoTokenizer
from src.pruning import prune_tokenizer_and_model

def test_prune_tokenizer_and_model_reduces_vocab_size():
    model_name = "sshleifer/tiny-gpt2"

    # We create a dummy dataset with very few tokens
    dataset = [{"text": "hello"}]

    output_dir = "./test-lfm-pruned"

    # Run the pruning function with min_freq=1
    # This means only tokens in "hello" and special tokens will be kept
    # So the vocab size should be significantly smaller than the original

    tokenizer_orig = AutoTokenizer.from_pretrained(model_name)
    orig_vocab_size = tokenizer_orig.vocab_size

    try:
        pruned_tokenizer, model = prune_tokenizer_and_model(
            model_name=model_name,
            dataset=dataset,
            min_freq=1,
            output_dir=output_dir
        )

        # Check that the vocabulary size is reduced
        assert pruned_tokenizer.vocab_size < orig_vocab_size
        assert model.config.vocab_size == pruned_tokenizer.vocab_size

        # Check embedding dimensions
        input_embeddings = model.get_input_embeddings().weight.data
        assert input_embeddings.shape[0] == pruned_tokenizer.vocab_size

        # We cannot guarantee the text 'hello' will encode properly if its constituent byte tokens were removed
        # but we can verify that the remaining vocab matches what was kept.
        assert pruned_tokenizer.vocab_size == 2 # special token and the combined word

    finally:
        # Cleanup
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        if os.path.exists("./temp_tokenizer_pruning"):
            shutil.rmtree("./temp_tokenizer_pruning")

if __name__ == "__main__":
    test_prune_tokenizer_and_model_reduces_vocab_size()
