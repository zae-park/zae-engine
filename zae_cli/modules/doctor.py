import time
import click
import platform
import psutil
import subprocess
import locale
import importlib


def check_os_info():
    os_info = platform.uname()
    return f"Operating System: {os_info.system} {os_info.release} {os_info.version} {os_info.machine}\nLocale: {locale.getdefaultlocale()}"


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
        return "[✗] NVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed."


def spinner_function(stop_event):
    spinner = ["-", "/", "|", "\\"]
    while not stop_event.is_set():
        for symbol in spinner:
            if stop_event.is_set():
                break
            click.echo(f"\r{symbol}", nl=False, flush=True)
            time.sleep(0.1)


def run_doctor(verbose=None):
    click.echo("Running zae doctor...\n")

    # Start spinner in a separate thread
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_function, args=(stop_event,))
    spinner_thread.start()

    # Perform checks
    os_info = check_os_info()
    click.echo(f"[✓] OS Info : {os_info.splitlines()[0]}")
    click.echo(f"    - {os_info.splitlines()[1]}")

    cpu_info = check_cpu_info()
    click.echo(f"[✓] CPU Info : {cpu_info.splitlines()[0]}")
    for line in cpu_info.splitlines()[1:]:
        click.echo(f"    {line}")

    torch_info = check_torch_info()
    click.echo(f"[✓] GPU Info : {torch_info.splitlines()[0]}")
    for line in torch_info.splitlines()[1:]:
        click.echo(f"    {line}")

    nvidia_smi_info = check_nvidia_smi(verbose=verbose)
    click.echo(f"[✓] PyTorch is installed.")
    click.echo(nvidia_smi_info)

    # Stop spinner
    stop_event.set()
    spinner_thread.join()

    click.echo("\nzae doctor check complete.")
