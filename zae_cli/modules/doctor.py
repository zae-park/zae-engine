import time
import click
import platform
import psutil
import subprocess
import locale
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def check_os_info():
    os_info = platform.uname()
    return (
        f"{os_info.system} {os_info.release} {os_info.version} ({os_info.machine})\nLocale: {locale.getdefaultlocale()}"
    )


def check_cpu_info():
    cpu_count = psutil.cpu_count(logical=False)
    logical_cpu_count = psutil.cpu_count(logical=True)
    cpu_info = platform.processor()
    cpu_details = f"{cpu_info} ({cpu_count} physical cores, {logical_cpu_count} logical cores)"
    if platform.system() == "Darwin" and "arm" in platform.machine().lower():
        cpu_details += "\nDetected Apple Silicon (M1, M2 series)"
    return cpu_details


def check_torch_info():
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        torch_version = torch.__version__
        details = [f"PyTorch is installed. Version: {torch_version}"]

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            details.append(f"{gpu_count} GPU(s) found")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                details.append(f"  [GPU {i}] : {gpu_name}")
                free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                details.append(f"    - Free Memory: {free_memory // (1024 * 1024)} MB")
                details.append(f"    - Total Memory: {total_memory // (1024 * 1024)} MB")

            cuda_version = torch.version.cuda
            cudnn_version = torch.backends.cudnn.version()
            details.append(f"CUDA Version: {cuda_version}")
            details.append(f"cuDNN Version: {cudnn_version}")
            if CUDA_HOME:
                details.append(f"CUDA_HOME: {CUDA_HOME}")
            else:
                details.append("CUDA_HOME: Not found")

        else:
            details.append("GPUs: No GPU available")

        return "\n".join(details)

    except ImportError:
        return "PyTorch is not installed."


def check_nvidia_smi():
    try:
        query_args = ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"]
        result = subprocess.run(query_args, capture_output=True, text=True)

        output = "NVIDIA SMI GPU Information:\n"
        gpu_info_lines = result.stdout.strip().split("\n")
        output += f"{len(gpu_info_lines)} device(s) found\n"
        for i, line in enumerate(gpu_info_lines):
            name, free_memory, total_memory = line.split(", ")
            output += f"  [GPU {i}] : {name}\n"
            output += f"    - Free Memory: {free_memory} MB\n"
            output += f"    - Total Memory: {total_memory} MB\n"

        return output
    except FileNotFoundError:
        return "NVIDIA SMI: Not found. Please ensure NVIDIA drivers are installed."


def spinner_function(stop_event):
    spinner = ["-", "/", "|", "\\"]
    while not stop_event.is_set():
        for symbol in spinner:
            if stop_event.is_set():
                break
            click.echo(f"\r{symbol}", nl=False, flush=True)
            time.sleep(0.1)


def run_doctor(verbose=False):
    click.echo("Running zae doctor...\n")

    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner_function, args=(stop_event,))
    spinner_thread.start()

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(check_os_info): "OS Info",
            executor.submit(check_cpu_info): "CPU Info",
            executor.submit(check_torch_info): "Torch Info",
            executor.submit(check_nvidia_smi): "GPU Info",
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if verbose:
                    click.echo(f"[✓] {futures[future]} :\n{result}")
                else:
                    click.echo(f"[✓] {futures[future]} : {result.splitlines()[0]}")
            except Exception as exc:
                click.echo(f"Error occurred: {exc}")

    stop_event.set()
    spinner_thread.join()

    click.echo("\nzae doctor check complete.")
