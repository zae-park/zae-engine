import unittest
import os
import shutil
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

    def test_doctor_command(self):
        """Test the 'doctor' command without --verbose."""
        result = self.runner.invoke(cli, ["doctor"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        # Add more assertions based on expected output from run_doctor

    def test_doctor_command_with_verbose(self):
        """Test the 'doctor' command with --verbose."""
        result = self.runner.invoke(cli, ["doctor", "--verbose"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Running zae doctor...", result.output)
        # Add more assertions based on expected verbose output from run_doctor


if __name__ == "__main__":
    unittest.main()
