import torch
from thop import profile


def analyze_model_memory(model_type: str, model_params: dict, model: torch.nn.Module):
    # bit width
    unit_bits = (
        model_params["quant_bits"]
        if model_params.get("is_qat", False)
        and model_params.get("do_int_forward", False)
        else 32
    )
    unit_bytes = unit_bits / 8

    # weights
    param_size = sum(p.numel() * unit_bytes for p in model.parameters()) / 1e3  # KB
    # activations
    buffer_size = sum(b.numel() * unit_bytes for b in model.buffers()) / 1e3  # KB

    # MACs estimation
    total_macs = 0
    if model_type == "lstm":
        seq_len = model_params["window_size"]
        input_size = model_params["num_in_features"]
        hidden_size = model_params["hidden_size"]
        num_layers = model_params.get("num_rnn_layers", 1)
        total_macs = 4 * hidden_size * (input_size + hidden_size) * seq_len * num_layers

    elif model_type == "transformer":
        seq_len = model_params["window_size"]
        d_model = model_params["d_model"]
        num_layers = model_params.get("num_enc_layers", 1)
        dim_feedforward = model_params.get("dim_feedforward", 4 * d_model)
        num_heads = model_params.get("nhead", 8)

        qkv_proj = 3 * seq_len * d_model * d_model
        attention_scores = seq_len * d_model * seq_len
        output_proj = seq_len * d_model * d_model
        ffn = 2 * seq_len * d_model * dim_feedforward

        layer_macs = qkv_proj + attention_scores + output_proj + ffn
        total_macs = layer_macs * num_layers

    weights_size = round(param_size, 2)
    activations_size = round(buffer_size, 2)
    total_macs = int(total_macs)
    return weights_size, activations_size, total_macs


def get_model_complexity(model_params: dict, model: torch.nn.Module, prefix: str):

    # determine the data type bit widths
    unit_bits = (
        model_params["quant_bits"]
        if model_params.get("is_qat", False)
        and model_params.get("do_int_forward", False)
        else 32  # default FP32 (32-bit)
    )
    unit_bytes = unit_bits / 8  # default FP32 (4 bytes)

    # calculate the parameters
    param_size = 0
    param_amount = 0
    param_tensors = 0
    for param in model.parameters():
        param_size += param.numel() * unit_bytes  # Calculate parameter size
        param_amount += param.numel()  # Calculate parameter amount
        param_tensors += 1  # Calculate parameter tensors
    param_size /= 1e3  # Convert to kilobytes(KB)

    # calculate the buffers
    buffer_size = 0
    buffer_amount = 0
    buffer_tensors = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * unit_bytes  # Calculate buffer size
        buffer_amount += buffer.numel()  # Calculate buffer amount
        buffer_tensors += 1  # Calculate buffer tensors
    buffer_size /= 1e3  # Convert to kilobytes(KB)

    # calculate the total model size
    model_size = param_size + buffer_size  # Calculate total model size

    is_qat = model_params.get("is_qat")
    do_int_forward = model_params.get("do_int_forward")

    if is_qat == False or (is_qat == True and do_int_forward == False):
        # If not using integer quantization
        memory_size = (
            torch.cuda.max_memory_allocated() / 1e6
        )  # Convert to megabytes(MB)

        # if model_params[]
        input_size = (
            model_params["batch_size"],
            model_params["window_size"],
            model_params["num_in_features"],
        )
        dummy_inputs = torch.randn(input_size).to(next(model.parameters()).device)
        flops, _ = profile(model, inputs=(dummy_inputs,), verbose=False)
        # flops /= 1e9  # Convert to gigaFLOPS(GFLOPS)
    else:
        memory_size = None
        flops = None

    return {
        f"{prefix}param_size (KB)": f"{param_size:.2f}",
        f"{prefix}param_amount": int(param_amount),
        f"{prefix}param_tensors": param_tensors,
        f"{prefix}buffer_size (KB)": f"{buffer_size:.2f}",
        f"{prefix}buffer_amount": int(buffer_amount),
        f"{prefix}buffer_tensors": buffer_tensors,
        f"{prefix}model_size (KB)": f"{model_size:.2f}",
        f"{prefix}memory_size (MB)": f"{memory_size:.2f}" if memory_size else None,
        f"{prefix}FLOPs": f"{flops:.2f}" if flops else None,
    }
