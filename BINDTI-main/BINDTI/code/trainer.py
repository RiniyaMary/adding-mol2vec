import os
import copy
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix,
    precision_recall_curve, precision_score
)

from models import binary_cross_entropy, cross_entropy_logits


class Trainer(object):
    def __init__(
        self,
        model, optim, device,
        train_dataloader, val_dataloader, test_dataloader,
        data_name, split, **config
    ):
        self.model = model
        self.optim = optim
        self.device = device

        # ---------------- CONFIG ----------------
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.n_class = config["DECODER"]["BINARY"]

        self.use_ld = config["SOLVER"]["USE_LD"]
        self.lr_decay = config["SOLVER"]["LR_DECAY"]
        self.decay_interval = config["SOLVER"]["DECAY_INTERVAL"]

        # Early stopping
        self.patience = config["SOLVER"].get("PATIENCE", 10)
        self.min_delta = config["SOLVER"].get("MIN_DELTA", 0.001)

        # Checkpoints
        self.checkpoint_interval = config["SOLVER"].get("CHECKPOINT_INTERVAL", 5)
        self.checkpoint_dir = config["RESULT"].get("CHECKPOINT_DIR", "../output/checkpoints/")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Output
        self.output_dir = config["RESULT"]["OUTPUT_DIR"] + f"{data_name}/{split}/"
        os.makedirs(self.output_dir, exist_ok=True)

        # ---------------- DATA ----------------
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        # ---------------- STATE ----------------
        self.current_epoch = 0
        self.resume_epoch = 0
        self.best_epoch = None
        self.best_val_loss = float("inf")
        self.best_auroc = 0
        self.epochs_no_improve = 0
        self.early_stop = False

        self.best_model = None

        # ---------------- LOGGING ----------------
        self.train_loss_epoch = []
        self.val_loss_epoch = []
        self.val_auroc_epoch = []
        self.test_metrics = {}

        # Pretty tables
        self.train_table = PrettyTable(["# Epoch", "Train_loss"])
        self.val_table = PrettyTable(["# Epoch", "AUROC", "AUPRC", "Val_loss"])
        self.test_table = PrettyTable([
            "# Best Epoch", "AUROC", "AUPRC", "F1",
            "Sensitivity", "Specificity", "Accuracy",
            "Threshold", "Test_loss"
        ])

        self.config = config

    # ==================================================
    # CHECKPOINTS
    # ==================================================
    def save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_auroc": self.best_auroc,
            "epochs_no_improve": self.epochs_no_improve,
            "train_loss_epoch": self.train_loss_epoch,
            "val_loss_epoch": self.val_loss_epoch,
            "val_auroc_epoch": self.val_auroc_epoch,
            "best_epoch": self.best_epoch
        }

        path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])

        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_auroc = checkpoint["best_auroc"]
        self.epochs_no_improve = checkpoint["epochs_no_improve"]
        self.train_loss_epoch = checkpoint["train_loss_epoch"]
        self.val_loss_epoch = checkpoint["val_loss_epoch"]
        self.val_auroc_epoch = checkpoint["val_auroc_epoch"]
        self.best_epoch = checkpoint["best_epoch"]
        self.resume_epoch = checkpoint["epoch"]

        # IMPORTANT SAFETY
        self.best_model = self.model

        print(f"🔄 Loaded checkpoint from epoch {self.resume_epoch}")
        return True

    # ==================================================
    # TRAIN LOOP
    # ==================================================
    def train(self, resume=False, checkpoint_path=None):
        start_epoch = 0
        if resume and checkpoint_path:
            if self.load_checkpoint(checkpoint_path):
                start_epoch = self.resume_epoch
                print(f"🔄 Resuming training from epoch {start_epoch}")

        for epoch in range(start_epoch, self.epochs):
            if self.early_stop:
                print(f"🎯 Early stopping at epoch {self.current_epoch}")
                break

            self.current_epoch = epoch + 1

            if self.use_ld and self.current_epoch % self.decay_interval == 0:
                self.optim.param_groups[0]["lr"] *= self.lr_decay

            train_loss = self.train_epoch()
            self.train_table.add_row(
                [f"epoch {self.current_epoch}", f"{train_loss:.4f}"]
            )
            self.train_loss_epoch.append(train_loss)

            auroc, auprc, val_loss = self.test(mode="val")
            self.val_table.add_row(
                [f"epoch {self.current_epoch}",
                 f"{auroc:.4f}", f"{auprc:.4f}", f"{val_loss:.4f}"]
            )
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            improvement = self.best_val_loss - val_loss
            if improvement > self.min_delta:
                self.best_val_loss = val_loss
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
                self.best_model = copy.deepcopy(self.model)
                self.epochs_no_improve = 0
                print("✅ Validation improved")
            else:
                self.epochs_no_improve += 1
                print(f"⏳ No improvement: {self.epochs_no_improve}/{self.patience}")

            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

            if self.current_epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(self.current_epoch)

        # Safety net
        if self.best_model is None:
            self.best_model = self.model
            self.best_epoch = self.current_epoch

        results = self.final_test()
        self.save_results()
        return results

    # ==================================================
    # TRAIN ONE EPOCH
    # ==================================================
    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for v_d, v_p, smiles_emb, labels in tqdm(self.train_dataloader):
            v_d = v_d.to(self.device)
            v_p = v_p.to(self.device)
            smiles_emb = smiles_emb.to(self.device)
            labels = labels.float().to(self.device)

            self.optim.zero_grad()
            _, _, _, score = self.model(v_d, v_p, smiles_emb)

            _, loss = binary_cross_entropy(score, labels)
            loss.backward()
            self.optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_dataloader)
        print(f"Training loss: {avg_loss:.4f}")
        return avg_loss

    # ==================================================
    # TEST / VALIDATION
    # ==================================================
    def test(self, mode="test"):
        loader = self.val_dataloader if mode == "val" else self.test_dataloader
        model = self.model if mode == "val" else self.best_model

        model.eval()
        y_true, y_pred = [], []
        total_loss = 0

        with torch.no_grad():
            for v_d, v_p, smiles_emb, labels in loader:
                v_d = v_d.to(self.device)
                v_p = v_p.to(self.device)
                smiles_emb = smiles_emb.to(self.device)
                labels = labels.float().to(self.device)

                _, _, _, score = model(v_d, v_p, smiles_emb)
                pred, loss = binary_cross_entropy(score, labels)

                total_loss += loss.item()
                y_true += labels.cpu().tolist()
                y_pred += pred.cpu().tolist()

        auroc = roc_auc_score(y_true, y_pred)
        auprc = average_precision_score(y_true, y_pred)
        avg_loss = total_loss / len(loader)

        if mode == "val":
            return auroc, auprc, avg_loss

        # ---------- Final test metrics ----------
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        precision = tpr / (tpr + fpr + 1e-8)
        f1 = 2 * precision * tpr / (precision + tpr + 1e-8)

        best_idx = np.argmax(f1[5:])
        threshold = thresholds[5:][best_idx]

        y_bin = (np.array(y_pred) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_bin)

        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
        precision1 = precision_score(y_true, y_bin)

        # CSV visualization
        pd.DataFrame({
            "y_label": y_true,
            "y_pred": y_pred
        }).to_csv("../output/visualization.csv", index=False)

        self.test_metrics = {
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1[5:][best_idx],
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "precision": precision1,
            "threshold": threshold,
            "test_loss": avg_loss,
            "best_epoch": self.best_epoch
        }

        self.test_table.add_row([
            self.best_epoch, f"{auroc:.4f}", f"{auprc:.4f}",
            f"{f1[5:][best_idx]:.4f}", f"{sensitivity:.4f}",
            f"{specificity:.4f}", f"{accuracy:.4f}",
            f"{threshold:.4f}", f"{avg_loss:.4f}"
        ])

        return (
            auroc, auprc, f1[5:][best_idx],
            sensitivity, specificity, accuracy,
            avg_loss, threshold, precision1
        )

    # ==================================================
    # SAVE RESULTS
    # ==================================================
    def save_results(self):
        torch.save(
            {
                "train_loss_epoch": self.train_loss_epoch,
                "val_loss_epoch": self.val_loss_epoch,
                "val_auroc_epoch": self.val_auroc_epoch,
                "test_metrics": self.test_metrics,
                "config": self.config
            },
            os.path.join(self.output_dir, "result_metrics.pt")
        )

        with open(os.path.join(self.output_dir, "train_table.txt"), "w") as f:
            f.write(self.train_table.get_string())

        with open(os.path.join(self.output_dir, "val_table.txt"), "w") as f:
            f.write(self.val_table.get_string())

        with open(os.path.join(self.output_dir, "test_table.txt"), "w") as f:
            f.write(self.test_table.get_string())

        print("💾 Results saved successfully")

    # ==================================================
    # FINAL TEST
    # ==================================================
    def final_test(self):
        return self.test(mode="test")
