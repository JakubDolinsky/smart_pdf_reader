"""
Integration tests for LlmClient with real Ollama and Mistral model.
Ollama is started/stopped automatically for the test class. Requires RUN_LLM_TESTS=True and model pulled.
For unit tests (mocked) see llm_client_unit_test.py.
Run directly: python AI_module/tests/infra_layer_tests/llm_client_integration_test.py
Or: python -m pytest AI_module/tests/infra_layer_tests/llm_client_integration_test.py -v
"""

import sys
from pathlib import Path

_file = Path(__file__).resolve()
_root = _file.parents[2]
if _root.name == "AI_module" and (_root.parent / "AI_module").is_dir():
    _root = _root.parent
_resolved_paths = [Path(p).resolve() for p in sys.path]
if _root not in _resolved_paths:
    sys.path.insert(0, str(_root))

import pytest
from AI_module.infra_layer.llm_client import LlmClient
from AI_module.config import RUN_LLM_TESTS, LLM_OLLAMA_HOST, LLM_OLLAMA_MODEL
from AI_module.infra_layer.ollama_lifecycle import managed as ollama_managed


def _ollama_unavailable() -> bool:
    if not RUN_LLM_TESTS:
        return True
    try:
        from ollama import Client
        client = Client(host=LLM_OLLAMA_HOST)
        resp = client.list()
        if hasattr(resp, "models"):
            names = [getattr(m, "model", "") for m in resp.models]
        elif isinstance(resp, dict):
            models = resp.get("models", [])
            names = [m.get("name", "") for m in models if isinstance(m, dict)]
        elif isinstance(resp, list):
            names = [getattr(m, "model", "") for m in resp]
        else:
            names = []
        return not any(
            n == LLM_OLLAMA_MODEL or n.startswith(LLM_OLLAMA_MODEL + ":") or LLM_OLLAMA_MODEL in n
            for n in names
        )
    except Exception:
        return True


@pytest.mark.skipif(not RUN_LLM_TESTS, reason="Real LLM tests skipped; set RUN_LLM_TESTS=True in config to run")
class TestLlmClientRealLLM:
    @pytest.fixture(scope="class")
    def ollama_for_real_llm(self):
        with ollama_managed():
            if _ollama_unavailable():
                pytest.skip("Ollama or model not available; install Ollama and run: ollama pull mistral:7b-instruct")
            yield

    @pytest.fixture
    def real_client(self, ollama_for_real_llm):
        return LlmClient()

    def test_real_llm_returns_non_empty_answer(self, real_client):
        result = real_client.answer("What is the Slovak language? Answer in one sentence.")
        assert isinstance(result, str) and len(result) > 0 and result == result.strip()

    def test_real_llm_reacts_to_different_prompts(self, real_client):
        prompts = [
            "What is the main purpose of the document? Answer briefly.",
            "Summarize in one sentence: This text is about software testing.",
            "Context: The project uses Python. Question: In which language is the project?",
        ]
        for prompt in prompts:
            result = real_client.answer(prompt)
            assert isinstance(result, str) and len(result) > 0 and result == result.strip()

    def test_real_llm_handles_question(self, real_client):
        result = real_client.answer("What does the word \"document\" mean?")
        assert isinstance(result, str) and len(result) >= 10

    def test_real_llm_handles_unknown_topic_gracefully(self, real_client):
        result = real_client.answer(
            "Context: This is a text about weather. Question: What was the exact amount in euros?"
        )
        assert isinstance(result, str) and len(result) >= 1


def main():
    return sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    main()
