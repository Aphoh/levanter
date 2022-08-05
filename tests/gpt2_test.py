import dataclasses

import jax.numpy as jnp
from jax.random import PRNGKey

from levanter.models.gpt2 import *

def test_gradient_checkpointing():
    # ensure that gradient checkpointing doesn't change the output
    # (this is a regression test for a bug that caused the output to change)
    for num_blocks in [1, 2, 4, 8, 12]:
        config = Gpt2Config(seq_len=16, hidden_dim=64, num_layers=num_blocks, num_heads=8, gradient_checkpointing=False)
        config_checkpoint = dataclasses.replace(config, gradient_checkpointing=True)
        key = PRNGKey(0)

        vocab = Axis("vocab", 128)

        model = Gpt2LMHeadModel(vocab, config, key=key)
        model_checkpoint = Gpt2LMHeadModel(vocab, config_checkpoint, key=key)

        input_ids = jnp.arange(128, dtype=jnp.int32)

        a1 = model(input_ids, key)
        a2 = model_checkpoint(input_ids, key)

        assert jnp.all(jnp.isclose(a1, a2))


