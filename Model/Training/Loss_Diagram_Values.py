import matplotlib.pyplot as plt
import sys
import os
import matplotlib.ticker as mticker
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)
from Model.Logging.Logger import setup_logger
train_logger = setup_logger('train', r'C:\Users\didri\Desktop\LearnReflect VideoEnchancer\AI UNet-Architecture\Model\Logging\Model_performance_logg\Model_Training_logg.txt')



##TRAINING####
def plot_loss_curves_Training_script_epoches(loss_history_Epoches, out_path="loss_curves_training_epoches.png"):
    epochs_count = len(loss_history_Epoches["combined"])
    epochs = range(1, epochs_count + 1)
    print(f"Epochs Count: {epochs_count}, Total Loss Entries: {len(loss_history_Epoches['Total_loss_per_epoch'])}")


    if len(loss_history_Epoches["Total_loss_per_epoch"]) != epochs_count:
        print(f"Error: Mismatch between epochs ({epochs_count}) and Total_loss_per_epoch ({len(loss_history_Epoches['Total_loss_per_epoch'])}).")
        train_logger.error(f"Epoch Loss Mismatch: Expected {epochs_count}, Found {len(loss_history_Epoches['Total_loss_per_epoch'])}.")
        return  

    plt.figure(figsize=(12, 6))

    if len(loss_history_Epoches["l1"]) > 0:
        plt.plot(epochs, loss_history_Epoches["l1"], label="L1-loss", color="blue",linewidth=2)

    if len(loss_history_Epoches["spectral"]) > 0:
        plt.plot(epochs, loss_history_Epoches["spectral"], label="spectral-loss", color="green",linewidth=2)

    if len(loss_history_Epoches["combined"]) > 0:
        plt.plot(epochs, loss_history_Epoches["combined"], label="combined", color="yellow",linewidth=2)

    if len(loss_history_Epoches["avg_trainloss"]) > 0:
        plt.plot(epochs, loss_history_Epoches["avg_trainloss"], label="avg_trainloss", color="black",linewidth=2)

    plt.plot(epochs, loss_history_Epoches["Total_loss_per_epoch"], label="Total_loss_per_epoch", color="purple",linewidth=2)


    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Time (Epoch)")
    plt.legend()
    plt.grid(True)

    plt.xticks(range(1,epochs_count + 1))
    plt.xlim([1, epochs_count])
    plt.savefig(out_path, bbox_inches="tight")



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
    batch_count = len(loss_history_Batches["combined"])
    batch_range = range(1, batch_count + 1)
    print(f"Batch Count: {batch_count}, Combined Loss Entries: {len(loss_history_Batches['combined'])}")

    plt.figure(figsize=(12,6))

    if len(loss_history_Batches["l1"]) > 0:
        plt.plot(batch_range, loss_history_Batches["l1"], label="L1-loss", color="blue", linewidth=1)

   
    if len(loss_history_Batches["spectral"]) > 0:
        plt.plot(batch_range, loss_history_Batches["spectral"], label="spectral-loss", color="green", linewidth=1)

 
    plt.plot(batch_range, loss_history_Batches["combined"], label="combined-loss", color="purple", linewidth=1)

    plt.xlabel("Batch")
    plt.ylabel("Loss (log Scale)")

    plt.title("Loss Over Time (Batch)")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.xlim([1,batch_count])
    plt.savefig(out_path, bbox_inches="tight")
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
















####FINE-TUNING#####
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



    


















#####EVALUATION#####

#Generer et diagram for SDR, SIR og SAR etter evaluering.
def plot_loss_curves_evaluation(sdr_list, sir_list, sar_list, out_path="loss_curves_evaluation.png"):
    
    epochs = range(1, len(sdr_list) + 1)

    plt.figure(figsize=(12, 6))
    plt.plot(epochs, sdr_list, label="SDR", color="blue",linewidth=1)
    plt.plot(epochs, sir_list, label="SIR", color="green",linewidth=1)
    plt.plot(epochs, sar_list, label="SAR", color="purple",linewidth=1)

    plt.xlabel("Batch")
    plt.ylabel("Value")
    plt.title("SDR, SIR, and SAR Over Evaluation")
    plt.legend()
    plt.grid(True)
    plt.yscale("log")
    plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".svg", ".pdf"]:
        out_path = base + ".png"

    plt.savefig(out_path, bbox_inches="tight")
    plt.close()







