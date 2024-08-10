import os
import click
import platform
import psutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


@click.group()
def cli():
    pass


def print_header():
    print("My tor summary (to see all details. run zae doctor -v):")
    print(f"[-\\|/] zae-engine (Channel master. 0.3.1, on Microsoft Windows [Version 10.0.19044.2486], locale ko-KR)")
    print(f"[-\\|/] Window Version (Installed version of Windows is version 10 or higher)")  # ignore
    print(f"[-\\|/] NVIDIA / CPU / MPU")
    print(f"\t\t [-\\|/] Check cudnn & CUDA Version")
    print(f"[-\\|/] Connected Device NVIDIA 1 / CPU / MPU 2")

    print(f"- No Issue Found!")


def check_os_info():
    os_info = platform.uname()
    return f"Operating System: {os_info.system} {os_info.release} {os_info.version} {os_info.machine}\nLocale: {platform.locale.getdefaultlocale()}"


def check_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    cpu_info = platform.processor()
    cpu_details = f"CPU: {cpu_info} ({cpu_count} physical cores, {logical_cpu_count} logical cores)"
    if platform.system() == "Darwin" and "arm" in platform.machine().lower():
        cpu_details += "\nDetected Apple Silicon (M1, M2 series)"
    return cpu_details


def check_torch_info():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        torch_version = torch.__version__
        details = [f"[✓] PyTorch is installed. Version: {torch_version}"]

        # GPU information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            details.append(f"GPUs: {gpu_count} available")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                details.append(f"  - GPU {i}: {gpu_name}")

            # CUDA and cuDNN information
            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            details.append(f"CUDA Version: {cuda_version}")
            details.append(f"cuDNN Version: {cudnn_version}")
        else:
            details.append("GPUs: No GPU available")

        # Additional check for CUDA_HOME
        if CUDA_HOME:
            details.append(f"CUDA_HOME: {CUDA_HOME}")
        else:
            details.append("CUDA_HOME: Not found")

        return "\n".join(details)

    except ImportError:
        return "[✗] PyTorch is not installed."


def check_nvidia_smi(verbose=False):
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        output = f"NVIDIA SMI:\n{result.stdout}"
        if verbose:
            # Fetch additional details if verbose flag is set
            detailed_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv"],
                capture_output=True,
                text=True,
            )
            output += f"\nDetailed GPU Info:\n{detailed_result.stdout}"
        return output
    except FileNotFoundError:
        return "NVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed."


def spinner_function(stop_event):
    spinner = ["-", "/", "|", "\\"]
    while not stop_event.is_set():
        for symbol in spinner:
            if stop_event.is_set():
                break
            click.echo(f"\r{symbol}", nl=False)
            time.sleep(0.1)


@cli.command()
@click.option("--verbose", is_flag=True, help="Show detailed information for NVIDIA SMI.")
def doctor(verbose):
    click.echo("Running zae doctor...\n")

    # Start spinner in a separate thread
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_function, args=(stop_event,))
    spinner_thread.start()

    # Perform checks
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(check_os_info): "OS Info",
            executor.submit(check_cpu_info): "CPU Info",
            executor.submit(check_torch_info): "Torch Info",
            executor.submit(check_nvidia_smi, verbose=verbose): "NVIDIA SMI",
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                click.echo(f"\n{result}")
            except Exception as exc:
                click.echo(f"Error occurred: {exc}")

    # Stop spinner
    stop_event.set()
    spinner_thread.join()

    click.echo("\nzae doctor check complete.")


if __name__ == "__main__":
    cli()
