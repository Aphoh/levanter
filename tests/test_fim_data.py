import tempfile

import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax

from levanter.data.text import FIMUrlSourceConfig, mk_fim_dataset
from levanter.store.cache import CacheOptions
from tiny_test_corpus import write_fim_data


@pytest.fixture
def fim_config():
    """Standard FIM configuration for testing."""
    return {
        "add_router_token": False,
        "predict_router_token": False,
        "shuffle": True,
        "pack": True,
        "cache_options": CacheOptions.no_fanciness(batch_size=128),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("predict_prefix", [True, False])
@pytest.mark.parametrize("predict_fim_token", [True, False])
async def test_fim_prediction_masks(fim_config, predict_prefix, predict_fim_token):
    """Test that prediction masks are correctly applied based on configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        max_len = 128
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", length=4)

        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            predict_prefix=predict_prefix,
            predict_fim_token=predict_fim_token,
            **fim_config,
        )

        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", max_len)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0), await_finished=False)

        # Get a sample and verify prediction masks
        dset_len = await dataset.async_len()
        example = (await dataset.get_batch([dset_len - 1]))[0]

        # Get token positions
        fim_token_id, eos_token_id, prefix_token_id, pad_token_id = tokenizer.convert_tokens_to_ids(
            [cfg.middle_token, cfg.eos_token, cfg.prefix_token, tokenizer.pad_token]
        )

        prefix_positions = jnp.argwhere(example.tokens.array == prefix_token_id).flatten()
        fim_positions = jnp.argwhere(example.tokens.array == fim_token_id).flatten()
        eos_positions = jnp.argwhere(example.tokens.array == eos_token_id).flatten()
        pad_positions = jnp.argwhere(example.tokens.array == pad_token_id).flatten()

        loss_mask = example.loss_mask.array

        # Basic assertions
        assert not loss_mask[pad_positions].any().item(), "Should not predict from pad tokens"
        assert (
            len(prefix_positions) == len(fim_positions) == len(eos_positions)
        ), "Should have equal numbers of special tokens"

        # Test prediction masks for each FIM sequence
        for prefix_pos, fim_pos, eos_pos in zip(prefix_positions, fim_positions, eos_positions):
            # Check prefix prediction
            prefix_mask = loss_mask[prefix_pos : fim_pos - 1]
            assert prefix_mask.all().item() == predict_prefix, f"Prefix prediction mismatch: expected {predict_prefix}"

            # Check FIM token prediction
            fim_mask = loss_mask[fim_pos - 1].item()
            assert fim_mask == predict_fim_token, f"FIM token prediction mismatch: expected {predict_fim_token}"

            # Check completion prediction (should always be True)
            completion_mask = loss_mask[fim_pos:eos_pos]
            assert completion_mask.all().item(), "Should always predict completion"

            # Check that we don't predict after EOS
            assert not loss_mask[eos_pos].item(), "Should not predict after EOS"


@pytest.mark.asyncio
async def test_router_token_functionality():
    """Test router token addition and prediction masking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", length=4)

        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            add_router_token=True,
            predict_router_token=True,
            shuffle=False,
            pack=False,
            cache_options=CacheOptions.no_fanciness(batch_size=128),
        )

        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", 128)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0), await_finished=False)

        example = (await dataset.get_batch([0]))[0]

        # Verify router token is present and router indices are set correctly
        assert example.router_hs_idxs is not None, "Router indices should be set when router token is enabled"
        assert example.router_input_mask is not None, "Router input mask should be set when router token is enabled"


@pytest.mark.asyncio
async def test_tokenizer_special_tokens():
    """Test that all required special tokens are properly added to tokenizer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", length=2)

        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            add_router_token=True,
        )

        tokenizer = cfg.the_tokenizer

        # Check all required tokens are in tokenizer
        required_tokens = [
            cfg.prefix_token,
            cfg.middle_token,
            cfg.suffix_token,
            cfg.file_sep_token,
            cfg.eos_token,
            cfg.pad_token,
            cfg.router_token,
        ]

        valid_tokens = [a.content for a in tokenizer.added_tokens_decoder.values()]
        for token in required_tokens:
            assert token in valid_tokens, f"Token {token} not found in tokenizer"


def test_illegal_char_replacement():
    """Test that illegal characters are properly replaced in content."""
    cfg = FIMUrlSourceConfig(middle_token="<|middle|>", eos_token="<eos>", replacer_trim_chars="<>|")

    illegal_str = f"some {cfg.middle_token} illegal {cfg.eos_token} chars"
    cleaned_str = cfg.replace_illegal_chars(illegal_str)

    # Verify original tokens are replaced
    assert cfg.middle_token not in cleaned_str
    assert cfg.eos_token not in cleaned_str

    # Verify content is preserved
    assert "some" in cleaned_str
    assert "illegal" in cleaned_str
    assert "chars" in cleaned_str

    # Verify cleaned tokens are present
    assert "<|cleaned_middle|>" in cleaned_str
    assert "<cleaned_eos>" in cleaned_str


@pytest.mark.asyncio
async def test_file_separator_inclusion():
    """Test file separator token inclusion when configured."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", length=2)

        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            always_include_file_sep=True,
            add_router_token=False,
            shuffle=False,
        )

        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", 256)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0), await_finished=False)

        example = (await dataset.get_batch([0]))[0]
        decoded = tokenizer.decode(example.tokens.array)

        # Should contain file separator token
        assert (
            cfg.file_sep_token in decoded
        ), "File separator token should be present when always_include_file_sep=True"


@pytest.mark.asyncio
async def test_empty_source_handling():
    """Test handling of empty or missing data sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[],  # Empty URLs
        )

        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", 128)

        with pytest.raises(ValueError, match="No source URLs configured"):
            mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0))


@pytest.mark.asyncio
async def test_dataset_basic_functionality():
    """Test basic dataset creation and data loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_data_jsonl = "file://" + write_fim_data(tmpdir + "/test_data.jsonl", length=10)

        cfg = FIMUrlSourceConfig(
            cache_dir=tmpdir + "/cache",
            train_urls=[test_data_jsonl],
            add_router_token=False,
            shuffle=False,
            pack=False,
        )

        tokenizer = cfg.the_tokenizer
        Pos = hax.Axis("Pos", 128)
        dataset = mk_fim_dataset(cfg, "train", tokenizer, Pos, key=PRNGKey(0))

        # Test dataset length and batch loading
        dset_len = await dataset.async_len()
        assert dset_len > 0, "Dataset should contain examples"

        # Test batch retrieval
        batch = await dataset.get_batch([0, min(1, dset_len - 1)])
        assert len(batch) <= 2, "Batch size should match request"

        # Test example structure
        example = batch[0]
        assert hasattr(example, "tokens"), "Example should have tokens"
        assert hasattr(example, "loss_mask"), "Example should have loss_mask"
        assert example.tokens.array.shape == (Pos.size,), f"Tokens should have shape {(Pos.size,)}"
