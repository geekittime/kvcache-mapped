"""Independent teacher-forcing check across the validation decoder chunk boundary."""
import sys
from pathlib import Path
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from validate import decoder


@torch.inference_mode()
def test_decoder_prediction_positions_and_chunk_boundary():
    torch.manual_seed(318)
    config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,pad_token_id=0,eos_token_id=96)
    config._attn_implementation='sdpa'
    model=Qwen3ForCausalLM(config).eval()
    prefix=torch.randint(1,96,(13,)).tolist()
    continuation=torch.randint(1,96,(131,)).tolist()
    # Independent full sequence: logit at prefix[-1] predicts continuation[0].
    direct=model(torch.tensor([prefix+continuation]),use_cache=False).logits[0]
    expected=direct[len(prefix)-1:len(prefix)-1+len(continuation)].float().log_softmax(-1)
    prefix_cache=model.model(torch.tensor([prefix[:-1]]),use_cache=True).past_key_values
    got=decoder(model,prefix_cache,prefix,continuation)
    assert got.shape==(131,97)
    torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)
    # Includes both boundaries 63→64 and127→128, not only an initial short chunk.
    assert torch.equal(got.argmax(-1),expected.argmax(-1))
