import unittest
import os
import shutil
from unittest.mock import patch
from click.testing import CliRunner
from zae_cli.main import cli  # Import the CLI function


class TestCLI(unittest.TestCase):

    def setUp(self):
        self.runner = CliRunner()
        self.test_file_path = "test_example.py"

    def tearDown(self):
        # Clean up any files created during the tests
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_hello_command(self):
        """Test the 'hello' command prints the correct message."""
        result = self.runner.invoke(cli, ["hello"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("My name is zae-park", result.output)

    def test_tree_command(self):
        """Test the 'tree' command. Note: Adjust this if you have specific expectations."""
        result = self.runner.invoke(cli, ["tree"])
        self.assertEqual(result.exit_code, 0)
        # Check if the command outputs anything specific. Update the assertion as needed.

    def test_example_command_with_path(self):
        """Test the 'example' command with the --path option."""
        result = self.runner.invoke(cli, ["example", "--path", self.test_file_path])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(os.path.isfile(self.test_file_path))
        with open(self.test_file_path, "r") as file:
            content = file.read()
            self.assertIn("Generated snippet file at", result.output)  # Update this to match actual output

    def test_example_command_without_path(self):
        """Test the 'example' command without the --path option."""
        result = self.runner.invoke(cli, ["example"])
        self.assertEqual(result.exit_code, 0)
        default_path = os.path.join(os.getcwd(), "zae_example.py")
        self.assertTrue(os.path.isfile(default_path))
        os.remove(default_path)  # Clean up

    @patch("zae_cli.modules.check_nvidia_smi")
    @patch("zae_cli.modules.check_torch_info")
    def test_doctor_command(self, mock_check_torch_info, mock_check_nvidia_smi):
        """Test the 'doctor' command without --verbose."""
        # Mock responses
        mock_check_torch_info.return_value = "[✗] PyTorch is not installed."
        mock_check_nvidia_smi.return_value = "NVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed."

        result = self.runner.invoke(cli, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        self.assertIn("[✗] PyTorch is not installed.", result.output)
        self.assertIn("NVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed.", result.output)

    @patch("zae_cli.modules.check_nvidia_smi")
    @patch("zae_cli.modules.check_torch_info")
    def test_doctor_command_with_verbose(self, mock_check_torch_info, mock_check_nvidia_smi):
        """Test the 'doctor' command with --verbose."""
        # Mock responses
        mock_check_torch_info.return_value = "[✓] PyTorch is installed. Version: 2.0.1+cu117\nCUDA Version: 11.7\nCUDA_HOME: /usr/local/cuda\ncuDNN Version: 8500"
        mock_check_nvidia_smi.return_value = (
            "NVIDIA SMI:\nGPU 0: NVIDIA GeForce RTX 3070\n  Free Memory: 7145 MB\n  Total Memory: 8192 MB"
        )

        result = self.runner.invoke(cli, ["doctor", "--verbose"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        self.assertIn("[✓] PyTorch is installed.", result.output)
        self.assertIn("CUDA Version: 11.7", result.output)
        self.assertIn("NVIDIA SMI:\nGPU 0: NVIDIA GeForce RTX 3070", result.output)


if __name__ == "__main__":
    unittest.main()
