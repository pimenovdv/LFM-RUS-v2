import torch
import math

def test_vectorized():
    # Example logic to replace the loop
    valid_context = torch.tensor([10, 20, 10, 30, 20])
    unique_tokens = torch.tensor([10, 20, 30])

    # We want distances from the end of valid_context for each unique token.
    # The last index of each token.

    # We can do:
    seq_len = valid_context.numel()

    # Create a mask of matches
    # Shape: (num_unique, seq_len)
    matches = valid_context.unsqueeze(0) == unique_tokens.unsqueeze(1)

    # Find the last index. Since argmax finds the first, we flip the matches.
    # flipped matches:
    flipped_matches = matches.flip(dims=[1])
    # The first True in flipped_matches corresponds to the last True in original
    first_flipped_indices = flipped_matches.float().argmax(dim=1)

    # distances from the end is just the index in the flipped array!
    # Because if it's the last element, it's at index 0 in flipped array.
    distances = first_flipped_indices.float()

    print(distances)

test_vectorized()
