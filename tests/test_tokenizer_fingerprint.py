from models.tokenizer_fingerprint import tokenizer_fingerprint


class FakeBackend:
    def __init__(self, state):
        self.state = state

    def to_str(self):
        return self.state


class FakeTokenizer:
    def __init__(self, backend_state, special_tokens_map, chat_template):
        self.backend_tokenizer = FakeBackend(backend_state)
        self.special_tokens_map = special_tokens_map
        self.chat_template = chat_template
        self.add_bos_token = True
        self.add_eos_token = False
        self.model_max_length = 512
        self.padding_side = "right"
        self.truncation_side = "right"


def test_tokenizer_fingerprint_is_stable_for_mapping_order():
    first = FakeTokenizer(
        backend_state='{"model":"same"}',
        special_tokens_map={"eos_token": "</s>", "bos_token": "<s>"},
        chat_template="{{ messages }}",
    )
    second = FakeTokenizer(
        backend_state='{"model":"same"}',
        special_tokens_map={"bos_token": "<s>", "eos_token": "</s>"},
        chat_template="{{ messages }}",
    )

    assert tokenizer_fingerprint(first) == tokenizer_fingerprint(second)


def test_tokenizer_fingerprint_changes_with_tokenizer_behavior():
    reference = FakeTokenizer(
        backend_state='{"model":"first"}',
        special_tokens_map={"bos_token": "<s>"},
        chat_template="{{ messages }}",
    )
    changed_backend = FakeTokenizer(
        backend_state='{"model":"second"}',
        special_tokens_map={"bos_token": "<s>"},
        chat_template="{{ messages }}",
    )
    changed_template = FakeTokenizer(
        backend_state='{"model":"first"}',
        special_tokens_map={"bos_token": "<s>"},
        chat_template="{{ messages[0] }}",
    )

    reference_signature = tokenizer_fingerprint(reference)
    assert tokenizer_fingerprint(changed_backend) != reference_signature
    assert tokenizer_fingerprint(changed_template) != reference_signature
