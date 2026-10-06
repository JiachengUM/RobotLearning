import argparse
import time

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor


MODEL_ID = "openvla/openvla-7b"
DEVICE = "mps"
DTYPE = torch.bfloat16


def mps_memory_gb():
    if not torch.backends.mps.is_available():
        return None

    return torch.mps.current_allocated_memory() / (1024 ** 3)


def create_test_image():
    """
    创建一张简单的 synthetic image。

    它不是一个真实机器人 observation，
    只是为了验证整个 VLA pipeline 能不能跑通。
    """
    image = Image.new(
        "RGB",
        (224, 224),
        (210, 210, 210),
    )

    draw = ImageDraw.Draw(image)

    # 简单画一张“桌子”
    draw.rectangle(
        [0, 140, 224, 224],
        fill=(155, 110, 75),
    )

    # 一个红色物体
    draw.rectangle(
        [85, 95, 135, 145],
        fill=(220, 40, 40),
    )

    return image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a real robot image.",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="pick up the red object",
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. 确认 MPS
    # --------------------------------------------------

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is not available. "
            "Check PyTorch installation and arm64 Python."
        )

    print("Device:", DEVICE)
    print("dtype :", DTYPE)

    # --------------------------------------------------
    # 2. 加载 processor
    # --------------------------------------------------

    print("\nLoading processor...")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    # --------------------------------------------------
    # 3. 加载 OpenVLA
    # --------------------------------------------------

    print("Loading OpenVLA-7B...")
    start = time.time()

    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,

        # OpenVLA 官方模型包含自定义 HF code
        trust_remote_code=True,

        # 直接使用 BF16，约 2 bytes / parameter
        torch_dtype=DTYPE,

        # 减少加载时 CPU RAM 峰值
        low_cpu_mem_usage=True,

        # Mac 不使用 CUDA FlashAttention
        attn_implementation="eager",

        # 非常重要：
        # 直接尝试把模型加载到 MPS，
        # 避免先完整 load 到 CPU 再复制到 GPU。
        device_map=DEVICE,
    )

    vla.eval()

    print(
        f"Model loaded in {time.time() - start:.1f} s"
    )

    mem = mps_memory_gb()

    if mem is not None:
        print(
            f"PyTorch MPS allocated: {mem:.2f} GB"
        )

    # --------------------------------------------------
    # 4. 准备 image
    # --------------------------------------------------

    if args.image is None:
        print(
            "\nNo image supplied; "
            "using synthetic smoke-test image."
        )

        image = create_test_image()

    else:
        print(
            f"\nLoading image: {args.image}"
        )

        image = Image.open(
            args.image
        ).convert("RGB")

    print("Image size:", image.size)

    # --------------------------------------------------
    # 5. 构造 OpenVLA prompt
    # --------------------------------------------------

    instruction = args.instruction.lower()

    prompt = (
        "In: What action should the robot take to "
        f"{instruction}?\n"
        "Out:"
    )

    print("\nPrompt:")
    print(prompt)

    # --------------------------------------------------
    # 6. Processor:
    #    image + text -> tensors
    # --------------------------------------------------

    inputs = processor(
        prompt,
        image,
    )

    print("\nProcessor output:")

    for name, tensor in inputs.items():
        if hasattr(tensor, "shape"):
            print(
                name,
                tensor.shape,
                tensor.dtype,
            )

    # --------------------------------------------------
    # 7. tensors -> MPS
    # --------------------------------------------------

    inputs = inputs.to(
        DEVICE,
        dtype=DTYPE,
    )

    # --------------------------------------------------
    # 8. VLA inference
    # --------------------------------------------------

    print("\nRunning inference...")

    start = time.time()

    with torch.inference_mode():
        action = vla.predict_action(
            **inputs,

            # 使用 BridgeData V2 的 action statistics
            # 把 normalized action 转回真实 control scale
            unnorm_key="bridge_orig",

            # greedy / deterministic decoding
            do_sample=False,
        )

    elapsed = time.time() - start

    # --------------------------------------------------
    # 9. 输出 7-DoF action
    # --------------------------------------------------

    print(
        f"\nInference took {elapsed:.2f} s"
    )

    print("\n7-DoF action:")
    print(action)

    labels = [
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_roll",
        "delta_pitch",
        "delta_yaw",
        "gripper",
    ]

    print("\nDecoded:")

    for name, value in zip(
        labels,
        action,
    ):
        print(
            f"{name:12s}: {value:+.6f}"
        )


if __name__ == "__main__":
    main()