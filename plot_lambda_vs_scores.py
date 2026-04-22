import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/neurips/test_detailed_logs_safty.csv")

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(df["lambda_1"], df["jnk3"], label="jnk3", marker="o", s=60, alpha=0.8)
ax.scatter(df["lambda_1"], df["herg"], label="HERG", marker="s", s=60, alpha=0.8)

ax.set_xlabel("Lambda 1 (jnk3 weight)", fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("jnk3 and herg Scores vs Lambda", fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/neurips/lambda_vs_jnk3_herg.png", dpi=150)
plt.show()
print("Plot saved to results/neurips/lambda_vs_jnk3_herg.png")

