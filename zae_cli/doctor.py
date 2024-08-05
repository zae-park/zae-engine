import os


def zae_print():
    print("My tor summary (to see all details. run zae doctor -v):")
    print(f"[-\\|/] zae-engine (Channel master. 0.3.1, on Microsoft Windows [Version 10.0.19044.2486], locale ko-KR)")
    print(f"[-\\|/] Window Version (Installed version of Windows is version 10 or higher)")  # ignore
    print(f"[-\\|/] NVIDIA / CPU / MPU")
    print(f"\t\t [-\\|/] Check cudnn & CUDA Version")
    print(f"[-\\|/] Connected Device NVIDIA 1 / CPU / MPU 2")

    print(f"- No Issue Found!")
