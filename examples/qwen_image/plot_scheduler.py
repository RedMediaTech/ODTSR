import matplotlib.pyplot as plt
import torch
from diffsynth.schedulers import FlowMatchScheduler

# scheduler 初始化
scheduler = FlowMatchScheduler(
    sigma_min=0, sigma_max=1,
    extra_one_step=True,
    exponential_shift=True,
    exponential_shift_mu=0.8,
    shift_terminal=0.02
)
scheduler.set_timesteps(1000, training=True)
sigmas = scheduler.sigmas.cpu().numpy()

# 要标记的点
markers = [0, 250, 500, 750, 999]

plt.figure(figsize=(10, 5))
plt.plot(sigmas)

# 仅标记点，不显示数字
for i in markers:
    plt.scatter(i, sigmas[i], s=50, color='red')
    print(i, sigmas[i])

plt.ylabel("t value")
plt.title("Scheduler Curve")
plt.grid(True)
plt.xticks([])  # 横坐标完全隐藏
plt.tight_layout()

# 保存到硬盘
plt.savefig("scheduler_sigmas.png", dpi=300)
plt.close()
