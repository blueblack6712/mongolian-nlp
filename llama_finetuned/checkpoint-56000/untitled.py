import json
import matplotlib.pyplot as plt
import pandas as pd

# ====== LOAD TRAINING METRICS ======
def plot_training_metrics():
    # Load trainer state
    with open("./llama_finetuned/trainer_state.json", "r") as f:
        trainer_state = json.load(f)
    
    log_history = trainer_state["log_history"]
    df = pd.DataFrame(log_history)

    # Split into training and evaluation logs
    train_entries = [entry for entry in log_history if "loss" in entry]
    eval_entries = [entry for entry in log_history if "eval_loss" in entry]
    
    # Plot Training Loss
    plt.figure(figsize=(10, 5))
    steps = [entry["step"] for entry in train_entries]
    loss = [entry["loss"] for entry in train_entries]
    plt.plot(steps, loss, label="Training Loss", marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Steps")
    plt.grid(True)
    plt.legend()
    plt.savefig("training_loss.png")
    plt.close()

    # Plot Evaluation Loss
    if eval_entries:
        plt.figure(figsize=(10, 5))
        eval_steps = [entry["step"] for entry in eval_entries]
        eval_loss = [entry["eval_loss"] for entry in eval_entries]
        plt.plot(eval_steps, eval_loss, label="Evaluation Loss", color="red", marker="s")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Evaluation Loss Over Steps")
        plt.grid(True)
        plt.legend()
        plt.savefig("evaluation_loss.png")
        plt.close()

    # Plot Learning Rate (if available)
    if "learning_rate" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(steps, df["learning_rate"].dropna(), label="Learning Rate", color="green")
        plt.xlabel("Training Step")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.legend()
        plt.savefig("learning_rate.png")
        plt.close()

# ====== DISPLAY LoRA CONFIGURATION ======
def print_lora_config():
    with open("./llama_finetuned/adapter_config.json", "r") as f:
        config = json.load(f)
    
    print("\n=== LoRA Configuration ===")
    print(f"LoRA Rank (r): {config['r']}")
    print(f"LoRA Alpha (α): {config['lora_alpha']}")
    print(f"LoRA Dropout: {config['lora_dropout']}")

# ====== EXECUTE ======
if __name__ == "__main__":
    plot_training_metrics()
    print_lora_config()
    print("Graphs saved as training_loss.png, evaluation_loss.png, and learning_rate.png")