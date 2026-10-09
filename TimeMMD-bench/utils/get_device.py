import torch
import pynvml
import time

def get_free_memory(devices):
    print('Getting free memory...')
    if not torch.cuda.is_available():
        return None

    pynvml.nvmlInit()
    # num_devices = torch.cuda.device_count()
    free_mem_info = []
    for i in devices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_ratio = mem_info.free / mem_info.total
        free_mem_info.append(free_ratio)
    pynvml.nvmlShutdown()


    try:
        # 找到空闲内存最多的 GPU
        max_idx = 0
        max_ratio = 0.0

        # 检查是否有 GPU 的空闲内存超过 86%
        for idx, ratio in enumerate(free_mem_info):
            print(ratio)
            if ratio > 0.99:
                device = torch.device(f"cuda:{idx}")
                print(f"Using GPU {idx} with {ratio * 100:.2f}% free memory.")
                break

            if ratio > max_ratio:
                max_ratio = ratio
                max_idx = idx

        # 如果没有 GPU 的空闲内存超过 86%，检查空闲内存最多的 GPU 是否大于 60%
        if max_ratio > 0.25: #  and free_mem_info[0] > 0.30
            device = max_idx
            print(
                f"No GPU has more than 25% free memory. Using GPU {max_idx} with {max_ratio * 100:.2f}% free memory.")
            # else:
            #     # 如果最大空闲内存小于 60%，程序陷入等待
            #     print("No GPU has sufficient free memory. Waiting for a GPU to become available...")
            #     time.sleep(1)  # 每隔 10 秒检查一次
            #     continue  # 跳过当前循环，重新检查

        return device
    except:
        print("No GPU has more than 25% free memory. ")
        return None