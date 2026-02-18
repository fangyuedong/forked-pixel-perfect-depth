"""
验证 model_to_internal 和 internal_to_model 是否互为逆运算

如果它们是真正的逆运算，那么：
internal_to_model(model_to_internal(x)) = x
"""
import torch


def model_to_internal(x, abt):
    """Model format -> Internal format"""
    factor = torch.sqrt(abt) + torch.sqrt(1 - abt + 1e-8)
    return x * factor


def internal_to_model_old(x_t, abt):
    """Old implementation with extra epsilon in denominator"""
    factor = torch.sqrt(abt) + torch.sqrt(1 - abt + 1e-8)
    return x_t / (factor + 1e-8)  # 多余的 epsilon


def internal_to_model_new(x_t, abt):
    """New implementation without extra epsilon"""
    factor = torch.sqrt(abt) + torch.sqrt(1 - abt + 1e-8)
    return x_t / factor


def test_conversion_reversibility():
    """测试格式转换的可逆性"""
    print("=" * 80)
    print("测试 model_to_internal 和 internal_to_model 的可逆性")
    print("=" * 80)

    # 测试不同的 abt 值
    test_abts = [
        0.0,   # 纯噪声
        0.1,   # 高噪声
        0.5,   # 中间状态
        0.9,   # 接近干净
        0.99,  # 几乎干净
    ]

    print("\n测试 1: 使用旧版 internal_to_model（带 +1e-8）\n")
    print(f"{'abt':<10} {'factor':<12} {'max_diff':<15} {'reversible'}")
    print("-" * 60)

    old_all_reversible = True
    for abt_val in test_abts:
        # 创建测试数据
        x = torch.randn(1, 1, 8, 8)

        # 进行往返转换
        abt_tensor = torch.tensor(abt_val)
        x_internal = model_to_internal(x, abt_tensor)
        x_back_old = internal_to_model_old(x_internal, abt_tensor)

        # 计算差异
        diff_old = torch.abs(x - x_back_old)
        max_diff_old = diff_old.max().item()

        # 计算 factor 值
        with torch.no_grad():
            factor = torch.sqrt(abt_tensor) + torch.sqrt(1 - abt_tensor + 1e-8)

        is_reversible = max_diff_old < 1e-6
        old_all_reversible = old_all_reversible and is_reversible

        status = "✓" if is_reversible else "✗"
        print(f"{abt_val:<10.4f} {factor.item():<12.6f} {max_diff_old:<15.10e} {status}")

    print("\n" + "=" * 80)
    print("\n测试 2: 使用新版 internal_to_model（不带 +1e-8）\n")
    print(f"{'abt':<10} {'factor':<12} {'max_diff':<15} {'reversible'}")
    print("-" * 60)

    new_all_reversible = True
    for abt_val in test_abts:
        # 创建测试数据
        x = torch.randn(1, 1, 8, 8)

        # 进行往返转换
        abt_tensor = torch.tensor(abt_val)
        x_internal = model_to_internal(x, abt_tensor)
        x_back_new = internal_to_model_new(x_internal, abt_tensor)

        # 计算差异
        diff_new = torch.abs(x - x_back_new)
        max_diff_new = diff_new.max().item()

        # 计算 factor 值
        with torch.no_grad():
            factor = torch.sqrt(abt_tensor) + torch.sqrt(1 - abt_tensor + 1e-8)

        is_reversible = max_diff_new < 1e-6
        new_all_reversible = new_all_reversible and is_reversible

        status = "✓" if is_reversible else "✗"
        print(f"{abt_val:<10.4f} {factor.item():<12.6f} {max_diff_new:<15.10e} {status}")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    if not old_all_reversible:
        print("✗ 旧版（line 259 带 +1e-8）：不可逆")
        print("  这解释了为什么测试之前失败")
        print()

    if new_all_reversible:
        print("✓ 新版（移除 line 259 的 +1e-8）：完全可逆")
        print("  理论上 format conversion 不改变值")
        print()

    if new_all_reversible:
        print("理论上：if self.n_steps > 0: 条件检查应该是不必要的")
        print("因为 internal_to_model(model_to_internal(x)) = x")
        print()
        print("但是：实际测试表明移除条件检查后失败")
        print("这说明问题可能出在其他地方，而不是格式转换本身")
        print()
        print("可能的原因：")
        print("1. abt 值在两次转换之间发生了变化")
        print("2. 数值精度问题累积")
        print("3. 其他隐藏的状态变化")

    print("=" * 80)

    return new_all_reversible


if __name__ == '__main__':
    test_conversion_reversibility()
