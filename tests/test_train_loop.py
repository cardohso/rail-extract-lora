"""Real torch and PEFT on a tiny random Granite model. Skipped in CI, which has no torch; run with
`uv run --frozen --group train pytest tests/test_train_loop.py`."""

import json
import string

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import GraniteConfig, GraniteForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from rail_extract import prompts, schema, train_lora  # noqa: E402
from rail_extract import training as T  # noqa: E402
from rail_extract.backends import HFBackend  # noqa: E402

TEMPLATE = (
    "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)
CFG = T.TrainConfig("tiny", r=2, lr=1e-3, epochs=3, batch_size=2, dropout=0.1)


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny")
    chars = sorted(set(string.printable))
    vocab = {"<unk>": 0, "<|im_end|>": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    core = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    core.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core, eos_token="<|im_end|>", pad_token="<|im_end|>", unk_token="<unk>"
    )
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    torch.manual_seed(0)
    config = GraniteConfig(
        vocab_size=len(vocab),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        pad_token_id=1,
        eos_token_id=1,
        bos_token_id=1,
    )
    GraniteForCausalLM(config).save_pretrained(path)
    return str(path)


@pytest.fixture
def short_prompt(monkeypatch):
    monkeypatch.setattr(prompts, "system_prompt", lambda variant="v1": "sys")


def rows():
    return [
        {"id": f"p{i}", "heading": "H", "text": f"text {i}", "gold": [], "stratum": "none"}
        for i in range(4)
    ]


def lora_weights(net):
    return {k: v.detach().cpu().clone() for k, v in net.named_parameters() if v.requires_grad}


def fresh(model_dir, short=True):
    tok, net, device = train_lora.load_for_training(CFG, model_dir, None, seed=1)
    examples, dropped = train_lora.encode(rows(), tok)
    assert not dropped and len(examples) == 4
    return tok, net, device, examples


def test_resumed_run_equals_uninterrupted_run(model_dir, short_prompt, tmp_path):
    tok, net, device, ex = fresh(model_dir)
    log = train_lora.train_loop(CFG, net, tok, ex, tmp_path / "a", 1, device)
    assert log["steps"] == 6 and len(log["losses"]) == 6
    full = lora_weights(net)

    tok, net, device, ex = fresh(model_dir)
    train_lora.train_loop(CFG, net, tok, ex, tmp_path / "b", 1, device, stop_after=3)
    tok, net, device, ex = fresh(model_dir)  # a new process after a crash
    log2 = train_lora.train_loop(CFG, net, tok, ex, tmp_path / "b", 1, device)
    resumed = lora_weights(net)

    assert len(log2["losses"]) == 6
    assert full.keys() == resumed.keys() and full
    for k in full:
        assert torch.equal(full[k], resumed[k]), k


def test_training_changes_only_lora_and_lowers_loss(model_dir, short_prompt, tmp_path):
    tok, net, device, ex = fresh(model_dir)
    before = {k: v.detach().clone() for k, v in net.named_parameters()}
    cfg = T.TrainConfig("tiny2", r=2, lr=5e-3, epochs=20, batch_size=2, dropout=0.0)
    log = train_lora.train_loop(cfg, net, tok, ex, tmp_path / "c", 1, device)
    assert sum(log["losses"][-4:]) < sum(log["losses"][:4])
    for k, v in net.named_parameters():
        if "lora_" in k:
            assert v.requires_grad
        else:
            assert not v.requires_grad and torch.equal(v, before[k]), k


def test_adapter_saves_and_loads_for_inference(model_dir, short_prompt, tmp_path):
    tok, net, device, ex = fresh(model_dir)
    train_lora.train_loop(CFG, net, tok, ex, tmp_path, 1, device)
    assert train_lora.save_adapter(net, tmp_path) > 0
    backend = HFBackend(
        model_dir, None, adapter=str(tmp_path / "adapter"), max_new_tokens=5, device="cpu"
    )
    out = backend.generate([[{"role": "user", "content": "hi"}]])
    assert len(out) == 1 and isinstance(out[0], str)


def test_overfit_step_scores_generated_output(model_dir, short_prompt, tmp_path):
    tok, net, device, ex = fresh(model_dir)
    train_lora.train_loop(CFG, net, tok, ex, tmp_path, 1, device)
    result = train_lora.run_overfit(net, tok, rows(), tmp_path, max_new_tokens=5)
    assert 0 <= result["schema_valid"] <= 1 and "f1" in result
    lines = (tmp_path / "overfit_predictions.jsonl").read_text().splitlines()
    assert len(lines) == 4 and json.loads(lines[0])["id"] == "p0"


def test_constrained_decoding_yields_schema_valid_json(model_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    net = AutoModelForCausalLM.from_pretrained(model_dir)
    backend = HFBackend.from_loaded(net, tok, constrained=True, max_new_tokens=600, batch_size=2)
    msgs = [{"role": "user", "content": "x"}]
    outs = backend.generate([msgs, msgs + [{"role": "user", "content": "longer"}]])
    # A random model under the grammar stays valid wherever it stops; if it finished, it parses.
    assert len(outs) == 2
    for text in outs:
        if text.rstrip().endswith("}"):
            assert schema.parse_output(text) is not None


def test_micro_batches_give_the_same_gradient_as_one_batch(model_dir, short_prompt):
    tok, net, device, ex = fresh(model_dir)
    # make lora_B non-zero so every LoRA weight has a real gradient
    for k, v in net.named_parameters():
        if "lora_B" in k:
            torch.nn.init.normal_(v, std=0.05)
    net.eval()  # no dropout: the two runs must see the same function
    pad, idx = tok.pad_token_id, [0, 1, 2, 3]
    grads, losses = {}, {}
    for micro in (1, 2, 4):
        net.zero_grad(set_to_none=True)
        losses[micro] = train_lora.accumulate(net, ex, idx, micro, pad, device)
        grads[micro] = {
            k: v.grad.detach().clone() for k, v in net.named_parameters() if v.requires_grad
        }
    for micro in (1, 2):
        assert abs(losses[micro] - losses[4]) < 1e-5
        for k in grads[4]:
            assert torch.allclose(grads[micro][k], grads[4][k], rtol=1e-3, atol=1e-6), (micro, k)


def test_validation_scoring_writes_predictions_and_results(
    model_dir, short_prompt, tmp_path, monkeypatch
):
    from rail_extract import evaluate as E

    monkeypatch.setattr(train_lora, "load_split", lambda split: rows())
    monkeypatch.setattr(E, "load_split", lambda split, final=False, root=None: rows())
    tok, net, device, ex = fresh(model_dir)
    train_lora.train_loop(CFG, net, tok, ex, tmp_path / "run", 1, device)
    report = train_lora.eval_on_val(net, tok, "tiny", 1, root=tmp_path)
    assert set(report) == {"L1-tiny-s1", "L2-tiny-s1"}
    for system in report:
        assert (tmp_path / "predictions" / system / "val.jsonl").exists()
        assert (tmp_path / "results" / f"{system}.val.json").exists()
