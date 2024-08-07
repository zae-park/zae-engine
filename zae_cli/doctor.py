import os
import click
import platform
import psutil
import subprocess


@click.group()
def cli():
    pass


def zae_print():
    print("My tor summary (to see all details. run zae doctor -v):")
    print(f"[-\\|/] zae-engine (Channel master. 0.3.1, on Microsoft Windows [Version 10.0.19044.2486], locale ko-KR)")
    print(f"[-\\|/] Window Version (Installed version of Windows is version 10 or higher)")  # ignore
    print(f"[-\\|/] NVIDIA / CPU / MPU")
    print(f"\t\t [-\\|/] Check cudnn & CUDA Version")
    print(f"[-\\|/] Connected Device NVIDIA 1 / CPU / MPU 2")

    print(f"- No Issue Found!")


@cli.command()
def doctor():
    click.echo("Running zae doctor...\n")

    # OS information
    os_info = platform.uname()
    click.echo(f"Operating System: {os_info.system} {os_info.release} {os_info.version} {os_info.machine}")
    click.echo(f"Locale: {platform.locale.getdefaultlocale()}\n")

    # CPU information
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    cpu_info = platform.processor()
    click.echo(f"CPU: {cpu_info} ({cpu_count} physical cores, {logical_cpu_count} logical cores)")

    # Check for Apple M1, M2 series chip
    if platform.system() == "Darwin" and "arm" in platform.machine().lower():
        click.echo("Detected Apple Silicon (M1, M2 series)")

    click.echo("\n")

    # Torch information
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        torch_version = torch.__version__
        click.echo(f"[✓] PyTorch is installed. Version: {torch_version}")

        # GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            click.echo(f"GPUs: {gpu_count} available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                click.echo(f"  - GPU {i}: {gpu_name}")
            # CUDA and cuDNN information
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            click.echo(f"CUDA Version: {cuda_version}")
            click.echo(f"cuDNN Version: {cudnn_version}")
        else:
            click.echo("GPUs: No GPU available")

        # Additional check for CUDA_HOME
        if CUDA_HOME:
            click.echo(f"CUDA_HOME: {CUDA_HOME}")
        else:
            click.echo("CUDA_HOME: Not found")

    except ImportError:
        click.echo("[✗] PyTorch is not installed.")

    # Checking GPU compatibility (NVIDIA SMI)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        click.echo(f"\nNVIDIA SMI:\n{result.stdout}")
    except FileNotFoundError:
        click.echo("\nNVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed.\n")

    click.echo("\nzae doctor check complete.")


if __name__ == "__main__":
    cli()
