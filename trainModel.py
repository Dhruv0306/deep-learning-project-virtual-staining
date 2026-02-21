import os
from datetime import datetime

from history_utils import save_history_to_csv, visualize_history
from training_loop import train


def main():
    model_dir = f"data\\E_Staining_DermaRepo\\H_E-Staining_dataset\\models_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Model directory: {model_dir}")

    val_dir = os.path.join(model_dir, "validation_images")
    os.makedirs(val_dir, exist_ok=True)
    print(f"Validation image directory: {val_dir}")

    epoch_size = int(input("Enter Epoch Size: "))
    num_epochs = int(input("Enter Number of Epochs: "))

    history, G_AB, G_BA, D_A, D_B = train(
        epoch_size=epoch_size,
        num_epochs=num_epochs,
        model_dir=model_dir,
        val_dir=val_dir,
    )

    visualize_history(history, model_dir=model_dir)
    save_history_to_csv(history, f"{model_dir}\\training_history.csv")

    return history, G_AB, G_BA, D_A, D_B


if __name__ == "__main__":
    main()
