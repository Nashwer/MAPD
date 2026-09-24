from types import SimpleNamespace

from mapd.environment.vllm_policy import VLLMStudentPolicy


class FakeTokenizer:
    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "rendered prompt"


class FakeEngine:
    def generate(self, prompts, params):
        assert prompts == ["rendered prompt"]
        assert params.max_tokens == 64
        assert params.temperature == 0.0
        assert params.logprobs == 1
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=" <search>query</search> ",
                        token_ids=[7, 8],
                        logprobs=[
                            {7: SimpleNamespace(logprob=-0.25)},
                            {8: SimpleNamespace(logprob=-0.5)},
                        ],
                    )
                ]
            )
        ]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_vllm_policy_renders_messages_and_returns_generated_text():
    tokenizer = FakeTokenizer()
    policy = VLLMStudentPolicy(FakeEngine(), tokenizer, FakeSamplingParams)
    messages = [{"role": "user", "content": "question"}]

    output = policy.generate(messages, max_new_tokens=64)

    assert output == " <search>query</search> "
    assert tokenizer.messages == messages
    assert policy.last_token_ids == [7, 8]
    assert policy.last_token_log_probs == [-0.25, -0.5]
