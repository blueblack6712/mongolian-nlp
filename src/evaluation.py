import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configuration
LOG_DIR = Path("/dev/shm/bert_finetuned/checkpoint-59880")
TRAINER_STATE_FILE = LOG_DIR / "trainer_state.json"
OUTPUT_DIR = Path("/root")  # Directory to save plots

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load training history
with open(TRAINER_STATE_FILE) as f:
    training_history = json.load(f)

# Extract metrics
log_history = training_history["log_history"]
metrics = {
    "train_loss": [], "eval_loss": [], "eval_f1": [],
    "eval_precision": [], "eval_recall": [], 
    "step": [], "learning_rate": []
}

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        metrics["train_loss"].append(entry["loss"])
        metrics["step"].append(entry["step"])
        metrics["learning_rate"].append(entry.get("learning_rate", 0))
    elif "eval_loss" in entry:
        metrics["eval_loss"].append(entry["eval_loss"])
        metrics["eval_f1"].append(entry["eval_f1"])
        metrics["eval_precision"].append(entry["eval_precision"])
        metrics["eval_recall"].append(entry["eval_recall"])

# Align evaluation steps
eval_steps = metrics["step"][:len(metrics["eval_loss"])]

# 1. Print Numerical Results
print("\n" + "="*70)
print(f"{'TRAINING REPORT':^70}")
print("="*70)

# Print evaluation metrics table
print("\nEvaluation Metrics:")
print(f"{'Step':<8} | {'Loss':<8} | {'F1':<6} | {'Precision':<9} | {'Recall':<6}")
print("-"*55)
for step, loss, f1, prec, rec in zip(eval_steps, metrics["eval_loss"],
                                   metrics["eval_f1"], metrics["eval_precision"],
                                   metrics["eval_recall"]):
    print(f"{step:<8} | {loss:<8.4f} | {f1:<6.3f} | {prec:<9.3f} | {rec:<6.3f}")

# Print key statistics
max_f1_idx = np.argmax(metrics["eval_f1"])
print("\n" + "-"*70)
print(f"{'Key Statistics':^70}")
print("-"*70)
print(f"Final Training Loss:     {metrics['train_loss'][-1]:.4f}")
print(f"Best Validation F1:      {max(metrics['eval_f1']):.4f} (Step {eval_steps[max_f1_idx]})")
print(f"Best Precision:          {max(metrics['eval_precision']):.4f}")
print(f"Best Recall:             {max(metrics['eval_recall']):.4f}")
print(f"Final Learning Rate:     {metrics['learning_rate'][-1]:.2e}")
print("="*70 + "\n")

# 2. Visualization and Saving
def save_plot(fig, filename):
    """Helper function to save a figure"""
    path = OUTPUT_DIR / filename
    fig.savefig(path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved plot to {path}")

# Loss Curves
fig1 = plt.figure(figsize=(10, 6))
plt.plot(metrics["step"], metrics["train_loss"], label="Training")
plt.plot(eval_steps, metrics["eval_loss"], label="Validation")
plt.title("Loss Progression", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(fig1, "loss_progression.png")

# F1 Score Development
fig2 = plt.figure(figsize=(10, 6))
plt.plot(eval_steps, metrics["eval_f1"], color="green")
plt.title("F1 Score Progression", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("F1 Score", fontsize=12)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.annotate(f'Max: {max(metrics["eval_f1"]):.3f}',
            xy=(eval_steps[max_f1_idx], max(metrics["eval_f1"])),
            xytext=(20, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", color='red'))
save_plot(fig2, "f1_score_progression.png")

# Precision-Recall Comparison
fig3 = plt.figure(figsize=(10, 6))
plt.plot(eval_steps, metrics["eval_precision"], label="Precision")
plt.plot(eval_steps, metrics["eval_recall"], label="Recall")
plt.title("Precision vs Recall", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
save_plot(fig3, "precision_recall_comparison.png")

# Learning Rate Schedule
fig4 = plt.figure(figsize=(10, 6))
plt.plot(metrics["step"], metrics["learning_rate"], color="purple")
plt.title("Learning Rate Schedule", fontsize=14)
plt.xlabel("Steps", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.grid(True, alpha=0.3)
save_plot(fig4, "learning_rate_schedule.png")

print("\nAll plots saved to /root/ directory")