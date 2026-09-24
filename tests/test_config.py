from mapd.config import _expand_env


def test_environment_expansion_supports_shell_style_defaults(monkeypatch):
    monkeypatch.delenv("MAPD_TEST_BUDGET", raising=False)
    assert _expand_env("${MAPD_TEST_BUDGET:-5.0}") == "5.0"
    monkeypatch.setenv("MAPD_TEST_BUDGET", "2.5")
    assert _expand_env("${MAPD_TEST_BUDGET:-5.0}") == "2.5"
