import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker

def plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path="loss_curves_training_epoches.png"):
    epochs_count = len(loss_history_Epoches["combined"])
    epochs = range(1, epochs_count + 1)

    plt.figure(figsize=(10,6))

 
    if len(loss_history_Epoches["l1"]) > 0:
        plt.plot(epochs, loss_history_Epoches["l1"], label="L1-loss", color="blue")

    if len(loss_history_Epoches["spectral"]) > 0:
        plt.plot(epochs, loss_history_Epoches["spectral"], label="spectral-loss", color="green")

 
    plt.plot(epochs, loss_history_Epoches["combined"], label="combined-loss", color="purple")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Epoch)")
    plt.legend()

    if epochs_count > 1:
        plt.xticks(range(1, epochs_count+1, max(1, epochs_count // 10)))


    plt.xlim([1, epochs_count])
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))


    if epochs_count > 0:
        Final_loss = loss_history_Epoches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs_count}: Phenomenal ({Final_loss:.4f})"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs_count}: Excellent! (loss={Final_loss:.4f})"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs_count}: Good (loss={Final_loss:.4f})"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs_count}: Keep going (loss={Final_loss:.4f})"
        else:
            status_text = f"Epoch {epochs_count}: Higher loss ({Final_loss:.4f})"
    else:
        status_text = "No data"

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))


    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()






def plot_loss_curves_Training_script_Batches(loss_history_Batches, out_path="loss_curves_training_batches.png"):

    import matplotlib.ticker as mticker

    batch_count = len(loss_history_Batches["combined"])
    batch_range = range(1, batch_count + 1)

    plt.figure(figsize=(10,6))

    # Plot L1
    if len(loss_history_Batches["l1"]) > 0:
        plt.plot(batch_range, loss_history_Batches["l1"], label="L1-loss", color="blue")

    # Plot spectral
    if len(loss_history_Batches["spectral"]) > 0:
        plt.plot(batch_range, loss_history_Batches["spectral"], label="spectral-loss", color="green")

    # Plot combined
    plt.plot(batch_range, loss_history_Batches["combined"], label="combined-loss", color="purple")

    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Batch)")
    plt.legend()

   
    plt.yscale("log")


    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

   
    if batch_count > 20:

        plt.xticks(range(1, batch_count+1, max(1, batch_count//10)))
    plt.xlim([1, batch_count])

    if batch_count > 0:
        Final_loss = loss_history_Batches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Batch {batch_count}: Phenomenal ({Final_loss:.4f})"
        elif Final_loss < 0.05:
            status_text = f"Batch {batch_count}: Excellent (loss={Final_loss:.4f})"
        elif Final_loss < 0.1:
            status_text = f"Batch {batch_count}: Good (loss={Final_loss:.4f})"
        elif Final_loss < 0.2:
            status_text = f"Batch {batch_count}: Keep going (loss={Final_loss:.4f})"
        else:
            status_text = f"Batch {batch_count}: Higher loss ({Final_loss:.4f})"
    else:
        status_text = "No data"

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))

  
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()





def plot_loss_curves_FineTuning_script_(loss_history_finetuning_epoches, out_path="loss_curves_finetuning_epoches.png"):
    epochs_count = len(loss_history_finetuning_epoches["combined"])
    epochs = range(1, epochs_count + 1)
    plt.figure(figsize=(10,6))


    if len(loss_history_finetuning_epoches["l1"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["l1"], label="L1-loss", color="blue")


    if len(loss_history_finetuning_epoches["spectral"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["spectral"], label="spectral-loss", color="green")

    if len(loss_history_finetuning_epoches["perceptual"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["perceptual"], label="l1-loss", color="red")

    if len(loss_history_finetuning_epoches["mse"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["mse"], label="mse-loss", color="black")

    if len(loss_history_finetuning_epoches["multiscale"]) > 0:
        plt.plot(epochs, loss_history_finetuning_epoches["multiscale"], label="multiscale-loss", color="orange")



    plt.plot(epochs, loss_history_finetuning_epoches["combined"], label="combined-loss", color="purple")

    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Batch)")
    plt.legend()


    Final_loss = loss_history_finetuning_epoches["combined"][-1] if len(loss_history_finetuning_epoches["combined"]) > 0 else 0
    if epochs > 0:
        Final_loss = loss_history_finetuning_epoches["combined"][-1]
        if Final_loss < 0.01:
            status_text = f"Epoch {epochs}: Phenomenal ({Final_loss:.4f})"
        elif Final_loss < 0.05:
            status_text = f"Epoch {epochs}: Excellent (loss={Final_loss:.4f})"
        elif Final_loss < 0.1:
            status_text = f"Epoch {epochs}: Good (loss={Final_loss:.4f})"
        elif Final_loss < 0.2:
            status_text = f"Epoch {epochs}: Keep going (loss={Final_loss:.4f})"
        else:
            status_text = f"Epoch {epochs}: Higher loss ({Final_loss:.4f})"
    else:
        status_text = "No data"

    plt.text(0.5, 0.5, status_text, fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))


    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path)
    plt.close()
