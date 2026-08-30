"""Checks profiler command registration and CSV extraction without requiring GPU counters."""
import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/profile_benchmarks.sh"
FAMILIES = ("vector_add", "transpose", "reduction", "gemm", "softmax", "conv2d",
            "rmsnorm", "swiglu", "rope", "embedding", "adamw", "global_norm",
            "cross_entropy", "causal_softmax", "attention")


class ProfileTests(unittest.TestCase):
    def write_fixture(self, path):
        # Synthetic parser fixture, never a performance measurement.
        with path.open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("Kernel Name", "Metric Name", "Metric Unit", "Metric Value"))
            for kernel in ("vector_add_naive_kernel", "vector_add_vectorized_kernel",
                           "vector_add_backward_kernel",
                           "causal_softmax_forward_kernel<256, false>",
                           "causal_softmax_backward_kernel<64>",
                           "rmsnorm_forward_kernel", "rmsnorm_backward_kernel",
                           "rope_forward_kernel", "rope_backward_kernel",
                           "swiglu_forward_kernel", "swiglu_backward_kernel",
                           "embedding_forward_kernel", "embedding_backward_kernel",
                           "adamw_step_kernel", "global_norm_partial_kernel",
                           "global_norm_finalize_kernel", "clip_gradients_kernel",
                           "cross_entropy_forward_kernel", "cross_entropy_backward_kernel",
                           "attention_pack_qkv_kernel", "attention_unpack_gradients_kernel"):
                writer.writerow((kernel, "gpu__time_duration.sum", "ns", "100"))

    def test_names_and_no_forward_backward_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.csv"
            self.write_fixture(fixture)
            result = subprocess.run(["bash", str(SCRIPT), "summarize", "vector_add",
                                     str(fixture), "1024"], capture_output=True, text=True, check=True)
            self.assertNotIn("Unknown", result.stdout)
            self.assertIn("BlockReduce", result.stdout)
            self.assertIn("RMSNorm backward", result.stdout)
            backward = next(line for line in result.stdout.splitlines()
                            if "vector_add_backward_kernel" in line)
            self.assertNotIn("1.00x", backward)

    def test_all_registration_and_visible_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            fixture = directory / "fixture.csv"
            self.write_fixture(fixture)
            fake_ncu = directory / "ncu"
            fake_ncu.write_text("""#!/usr/bin/env bash
set -eu
csv=""
while [[ "$1" != "/bin/true" ]]; do
    if [[ "$1" == "--log-file" ]]; then shift; csv="$1"; fi
    shift
done
shift
printf '%s\\n' "$1" >> "$CALLS"
if [[ "${FAIL_PROFILE:-0}" == "1" ]]; then
    printf 'synthetic profiling failure\\n' > "$csv"
    exit 2
fi
cp "$FIXTURE" "$csv"
""")
            fake_ncu.chmod(0o755)
            env = dict(os.environ, NCU=str(fake_ncu), BIN="/bin/true",
                       OUT_DIR=str(directory / "reports"), FIXTURE=str(fixture),
                       CALLS=str(directory / "calls"))
            result = subprocess.run(["bash", str(SCRIPT), "all"],
                                    capture_output=True, text=True, env=env, check=True)
            self.assertEqual((directory / "calls").read_text().splitlines(), list(FAMILIES))
            for family in FAMILIES:
                self.assertTrue((directory / "reports" / f"{family}_summary.txt").exists())
            failed = subprocess.run(["bash", str(SCRIPT), "rmsnorm", "8", "512"],
                                    capture_output=True, text=True, env=dict(env, FAIL_PROFILE="1"))
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("synthetic profiling failure", failed.stderr)
            self.assertIn("could not profile rmsnorm", failed.stderr)
            self.assertEqual(failed.stdout, "")


if __name__ == "__main__":
    unittest.main()
