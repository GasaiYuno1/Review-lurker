"""LLM-клиент без сети: разбор JSON, заглушка, кэш, конфиг."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurker.config import LLMConfig, load_config  # noqa: E402
from lurker.llm import FakeLLM, LLMClient, LLMError, parse_json  # noqa: E402


class ParseJsonTests(unittest.TestCase):
    def test_plain_and_fenced(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('```json\n{"a": [1, 2]}\n```'), {"a": [1, 2]})
        self.assertEqual(parse_json('Вот ответ: {"a": "б"} — готово'), {"a": "б"})
        self.assertEqual(parse_json('текст [1, 2] хвост'), [1, 2])

    def test_garbage(self):
        with self.assertRaises(LLMError):
            parse_json("не json")


class StripThinkingTests(unittest.TestCase):
    def test_strip(self):
        from lurker.llm import strip_thinking
        self.assertEqual(strip_thinking("<think>думаю {\"x\": 1}</think>\n{\"a\": 1}"), '{"a": 1}')
        self.assertEqual(strip_thinking("{\"a\": 1}"), '{"a": 1}')
        self.assertEqual(strip_thinking("ответ <think>хвост без закрытия"), "ответ")
        self.assertEqual(parse_json(strip_thinking("<think>x</think>```json\n{\"a\": 2}\n```")), {"a": 2})


class FakeLLMTests(unittest.TestCase):
    def test_callable_reply(self):
        llm = FakeLLM(lambda msgs: '{"n": %d}' % len(msgs))
        self.assertEqual(llm.chat_json([{"role": "user", "content": "x"}]), {"n": 1})
        self.assertEqual(llm.calls, 1)
        self.assertEqual(llm.history[0][0]["content"], "x")


class CacheTests(unittest.TestCase):
    def test_second_call_hits_cache(self):
        with tempfile.TemporaryDirectory() as d:
            client = LLMClient(LLMConfig(model="m"), cache_dir=d)
            n = {"real": 0}

            def fake_request(body):
                n["real"] += 1
                return "ответ"
            client._request = fake_request
            msgs = [{"role": "user", "content": "привет"}]
            self.assertEqual(client.chat(msgs), "ответ")
            self.assertEqual(client.chat(msgs), "ответ")
            self.assertEqual(client.chat(msgs, temperature=0.5), "ответ")  # другие параметры — другой ключ
            self.assertEqual(n["real"], 2)
            client2 = LLMClient(LLMConfig(model="m"), cache_dir=d)
            client2._request = fake_request
            self.assertEqual(client2.chat(msgs), "ответ")
            self.assertEqual(n["real"], 2)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_file(self):
        cfg = load_config()
        self.assertEqual(cfg.llm.api_key_env, "LLM_API_KEY")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("llm:\n  base_url: http://localhost:11434/v1\n  model: qwen\nembedding:\n  device: cuda\n", encoding="utf-8")
            cfg = load_config(p)
            self.assertEqual((cfg.llm.base_url, cfg.llm.model, cfg.embedding.device), ("http://localhost:11434/v1", "qwen", "cuda"))
            self.assertEqual(cfg.cache.dir, ".cache")
            p.write_text("llm:\n  modle: x\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(p)
        with self.assertRaises(FileNotFoundError):
            load_config(Path(d) / "nope.yaml")

    def test_env_override(self):
        old = os.environ.get("LURKER_LLM_MODEL")
        os.environ["LURKER_LLM_MODEL"] = "override"
        try:
            self.assertEqual(load_config().llm.model, "override")
        finally:
            if old is None:
                del os.environ["LURKER_LLM_MODEL"]
            else:
                os.environ["LURKER_LLM_MODEL"] = old


if __name__ == "__main__":
    unittest.main()


class ClientBehaviourTests(unittest.TestCase):
    """Троттлинг, авто-откат json_mode, Retry-After, ping — без сети (подменяем urlopen)."""

    def _client(self, **kw):
        return LLMClient(LLMConfig(model="m", base_url="http://x/v1", max_retries=1, **kw), cache_dir=None)

    def test_json_mode_fallback_on_400(self):
        import urllib.error
        client = self._client()
        seen = []

        def fake_request(body):
            seen.append(dict(body))
            if "response_format" in body:
                raise LLMError("HTTP 400 от http://x/v1/chat/completions: response_format not supported")
            return '{"a": 1}'
        client._request = fake_request
        self.assertEqual(client.chat_json([{"role": "user", "content": "q"}]), {"a": 1})
        self.assertEqual(len(seen), 2)
        self.assertIn("response_format", seen[0])
        self.assertNotIn("response_format", seen[1])
        self.assertFalse(client._json_ok)
        client.chat_json([{"role": "user", "content": "q2"}])
        self.assertEqual(len(seen), 3)
        self.assertNotIn("response_format", seen[2])

    def test_throttle_and_usage(self):
        import io
        import json as _json
        import urllib.request
        client = self._client(rate_limit_rpm=600)  # 0.1 с между запросами
        calls = []

        class Resp(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=None):
            calls.append(time.time())
            return Resp(_json.dumps({"choices": [{"message": {"content": "hi"}}],
                                     "usage": {"prompt_tokens": 7, "completion_tokens": 3}}).encode())
        orig = urllib.request.urlopen
        urllib.request.urlopen = fake_urlopen
        try:
            for _ in range(3):
                self.assertEqual(client.chat([{"role": "user", "content": "x"}], temperature=0.1 * len(calls)), "hi")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(client.calls, 3)
        self.assertEqual(client.usage, {"prompt_tokens": 21, "completion_tokens": 9})
        self.assertGreaterEqual(calls[-1] - calls[0], 0.18)

    def test_ping(self):
        import io
        import json as _json
        import urllib.error
        import urllib.request
        client = self._client()

        class Resp(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def ok_urlopen(req, timeout=None):
            self.assertTrue(req.full_url.endswith("/v1/models"))
            return Resp(_json.dumps({"data": [{"id": "m"}, {"id": "n"}]}).encode())

        def denied_urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", {}, io.BytesIO(b"no key"))
        orig = urllib.request.urlopen
        try:
            urllib.request.urlopen = ok_urlopen
            self.assertEqual(client.ping()["models"], ["m", "n"])
            urllib.request.urlopen = denied_urlopen
            p = client.ping()
            self.assertFalse(p["ok"])
            self.assertEqual(p["status"], 401)
        finally:
            urllib.request.urlopen = orig


import time  # noqa: E402
