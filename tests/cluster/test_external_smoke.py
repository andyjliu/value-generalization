"""Check real fork entry points and run fork suites outside root collection."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.fork_suite


@pytest.fixture
def fork_process(tmp_path):
    def run(fork, arguments, *, python=sys.executable):
        path = ROOT / "external" / fork
        assert path.is_dir(), "initialize submodules with git submodule update --init --recursive"
        env = os.environ.copy()
        for key in list(env):
            if key.endswith("API_KEY") or key.endswith("BASE_URL") or key in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}:
                env.pop(key)
        env.update({
            "PYTHONPATH": os.pathsep.join(map(str, [path, path / "src", path / "data_generation"])),
            "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "PERSONA_LOCAL_API": "true",
            "OPENAI_API_KEY": "test-not-a-credential", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
            "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "XDG_DATA_HOME": str(tmp_path / "data"), "XDG_CACHE_HOME": str(tmp_path / "cache"),
            "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
        })
        result = subprocess.run([str(python), "-B", *arguments], cwd=tmp_path, env=env,
                                text=True, capture_output=True, timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr
    return run


@pytest.mark.parametrize("fork,entry,args", [
    ("conflictscope", "src/evaluate_models.py", ["--help"]),
    ("persona_vectors", "run_pipeline.py", ["run", "--help"]),
    ("persona_vectors", "generate_vec.py", ["--help"]),
    ("weight-steering", "cs_task_vectors.py", ["--help"]),
])
def test_fork_entrypoint_help(fork_process, fork, entry, args):
    output = fork_process(fork, [str(ROOT / "external" / fork / entry), *args])
    assert "usage" in output.lower()
    if fork == "persona_vectors" and entry == "generate_vec.py":
        # Keep extraction's dependency smoke coverage after making its CLI and
        # filtering helpers lazy. Import once in pytest, outside fork unit suites.
        from sentence_transformers import SentenceTransformer

        assert callable(SentenceTransformer)


def _suite_arguments(fork, *tests):
    path = ROOT / "external" / fork
    return ["-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(path),
            "--confcutdir", str(path), *(str(path / test) for test in tests)]


@pytest.mark.fork_suite
@pytest.mark.parametrize("tests", [
    ("test_generate_prefill_pairs.py", "test_chat_template.py", "test_generate_vec_filter.py",
     "test_local_openai.py", "test_fulltext_protocol.py", "test_env_precedence.py"),
    ("test_judge.py",),
    ("test_response_reuse.py",),
], ids=["shared-imports", "judge", "response-reuse"])
def test_persona_fork_suite(fork_process, tests):
    # Keep files that stub sys.modules['config'] out of the shared process.
    fork_process("persona_vectors", _suite_arguments("persona_vectors", *tests))


@pytest.mark.fork_suite
def test_msm_fork_suite_and_entrypoint(fork_process):
    python = ROOT / ".venvs/msm/bin/python"
    if not python.exists():
        pytest.skip("needs the optional msm environment: run ./scripts/setup_env.sh --msm")
    fork_process("model_spec_midtraining", ["-m", "src.aft.generate_chat", "--help"], python=python)
    fork_process("model_spec_midtraining", _suite_arguments("model_spec_midtraining", "tests"), python=python)
