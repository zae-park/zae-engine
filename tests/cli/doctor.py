import unittest
from click.testing import CliRunner
from zae_cli.cli import cli_run
import platform
import subprocess


class TestZaeDoctor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.runner = CliRunner()

    def test_doctor_no_verbose(self):
        """Test the `zae doctor` command without verbose flag."""
        result = self.runner.invoke(cli_run, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        self.assertIn("Operating System:", result.output)
        self.assertIn("CPU:", result.output)
        self.assertIn("PyTorch is installed.", result.output) or self.assertIn(
            "PyTorch is not installed.", result.output
        )
        self.assertIn("NVIDIA SMI:", result.output) or self.assertIn("NVIDIA SMI: Not found", result.output)

    def test_doctor_verbose(self):
        """Test the `zae doctor` command with verbose flag."""
        result = self.runner.invoke(cli_run, ["doctor", "--verbose"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        self.assertIn("Operating System:", result.output)
        self.assertIn("CPU:", result.output)
        self.assertIn("PyTorch is installed.", result.output) or self.assertIn(
            "PyTorch is not installed.", result.output
        )

        # Check for detailed NVIDIA SMI output
        if platform.system() == "Windows":
            # On Windows, nvidia-smi might not be available by default
            self.assertIn("NVIDIA SMI: Not found", result.output) or self.assertIn("NVIDIA-SMI", result.output)
        else:
            # On Unix-like systems, verify detailed SMI output
            self.assertIn("NVIDIA-SMI", result.output)

    def test_doctor_spinner(self):
        """Test the spinner during the `zae doctor` command."""
        result = self.runner.invoke(cli_run, ["doctor"])

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        self.assertIn("[-\\|/]", result.output)  # Check for spinner characters in output


if __name__ == "__main__":
    unittest.main()
