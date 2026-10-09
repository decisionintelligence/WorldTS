import torch
import time
from ptflops import get_model_complexity_info


def evaluate_model_efficiency(model, test_loader, device, batch_size=1):
    """
    评估模型效率指标：参数数量、MACs、GPU内存占用、单样本推理时间

    Args:
        model: 要评估的模型
        test_loader: 测试数据加载器
        device: 设备 (cuda/cpu)
        batch_size: 批次大小，默认为1

    Returns:
        dict: 包含效率指标的字典
    """
    model.eval()

    # 获取一个batch的数据
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
        if i == 0:
            batch_x = batch_x[:batch_size].float().to(device)
            break

    # 1. 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())

    # 2. 计算MACs (使用ptflops)
    input_shape = tuple(batch_x.shape[1:])  # 去掉batch维度
    macs, params = get_model_complexity_info(
        model,
        input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )

    # 3. 计算GPU内存占用
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(batch_x)

    torch.cuda.synchronize()
    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # 转换为MB

    # 4. 计算推理时间
    # 预热
    warmup_iterations = 10
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(batch_x)
            torch.cuda.synchronize()

    # 正式测量时间
    num_iterations = 100
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(batch_x)
            torch.cuda.synchronize()

    end_time = time.time()
    avg_inference_time = (end_time - start_time) / num_iterations * 1000  # 转换为毫秒

    # 计算单样本推理时间
    single_sample_time = avg_inference_time / batch_size

    # 整理结果
    results = {
        "Total Parameters": f"{total_params:,}",
        "MACs": macs,
        "GPU Memory (MB)": f"{gpu_memory:.2f}",
        "Batch Inference Time (ms)": f"{avg_inference_time:.4f}",
        "Single Sample Time (ms)": f"{single_sample_time:.4f}",
        "Batch Size": batch_size,
        "Input Shape": str(batch_x.shape)
    }

    # 打印结果
    print("\n" + "=" * 50)
    print("Model Efficiency Metrics")
    print("=" * 50)
    for key, value in results.items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    # 保存结果到文件
    with open("./efficiency_metrics.txt", "w", encoding='utf-8') as f:
        f.write("Model Efficiency Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        f.write("=" * 50 + "\n")

    return results
