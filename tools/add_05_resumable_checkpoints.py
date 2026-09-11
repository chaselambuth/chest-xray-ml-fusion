# Add resumable checkpoint behavior to the Notebook 05 trial runner.

from pathlib import Path

import nbformat


NOTEBOOK_PATH = Path("Experiments/05_Xray_Hyperparameter_Augmentation.ipynb")

nb = nbformat.read(NOTEBOOK_PATH, as_version=4)


nb.cells[0].source = """# **Notebook 5: Bio_ClinicalBERT Residual Gated Fusion Hyperparameter and Augmentation Tuning**

**Description:** This notebook tunes the selected Bio_ClinicalBERT residual gated fusion model from Notebook 03 using the original full-image pipeline selected in Notebook 04. It keeps the same dataset split, label space, text encoder, image encoder, and residual gated fusion architecture, then runs targeted augmentation, learning-rate, dropout, weight-decay, and class-weight trials. Each trial writes a last checkpoint every epoch and a best checkpoint on validation macro F1 improvement, so interrupted trials can resume from the next unfinished epoch.

The goal of this notebook is to determine whether conservative tuning can improve validation macro F1 and held-out test performance over the Notebook 04 original full-image gated-fusion baseline. Its conclusion determines whether the tuned checkpoint should replace the Notebook 04 checkpoint before Notebook 06 generates the final Grad-CAM case-level interpretation.

---
"""

cell6 = nb.cells[6].source
if "CHECKPOINT_POLICY" not in cell6:
    cell6 = cell6.replace(
        "RUN_TEST_FOR_EACH_TRIAL = False\n",
        (
            "RUN_TEST_FOR_EACH_TRIAL = False\n"
            "CHECKPOINT_POLICY = (\n"
            "    \"last checkpoint saved every epoch; best checkpoint saved on validation macro F1 improvement\"\n"
            ")\n"
        ),
    )
if 'print("Checkpoint policy:", CHECKPOINT_POLICY)' not in cell6:
    cell6 += 'print("Checkpoint policy:", CHECKPOINT_POLICY)\n'
nb.cells[6].source = cell6

nb.cells[46].source = """def run_trial(config):
    trial_slug = config["trial_name"]
    trial_dir = OUTPUT_DIR / trial_slug
    trial_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = trial_dir / "best_checkpoint.pt"
    last_checkpoint_path = trial_dir / "last_checkpoint.pt"
    trial_config = {
        **config,
        "model_architecture": MODEL_ARCHITECTURE,
        "text_encoder": TEXT_ENCODER_NAME,
        "encoder_spec": TEXT_ENCODER_SPEC,
        "fusion_block": FUSION_BLOCK_NAME,
        "image_pipeline": IMAGE_PIPELINE_NAME,
        "image_pipeline_source": IMAGE_PIPELINE_SOURCE,
        "checkpoint_policy": CHECKPOINT_POLICY,
    }
    save_json(trial_config, trial_dir / "trial_config.json")

    train_loader, val_loader, test_loader = make_loaders(
        config["augmentation"],
        batch_size=config.get("batch_size", BATCH_SIZE),
    )
    model = BioClinicalBertGatedFusionClassifier(
        num_classes=num_classes,
        text_dropout=config["text_dropout"],
        classifier_dropout=config["classifier_dropout"],
        freeze_bert=config.get("freeze_bert", FREEZE_BERT),
        freeze_image_backbone=config.get("freeze_image_backbone", FREEZE_IMAGE_BACKBONE),
        cache_dir=HF_CACHE_DIR,
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(config["class_weight_power"]))
    optimizer = configure_optimizer(
        model,
        lr_text=config["lr_text"],
        lr_image=config["lr_image"],
        weight_decay=config["weight_decay"],
    )
    if config.get("scheduler", "constant") != "constant":
        raise ValueError(f"Unsupported scheduler: {config['scheduler']}")
    scheduler = build_scheduler(optimizer)

    def move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    best_metric = -math.inf
    best_epoch = None
    history = []
    start_epoch = 1
    resumed_from_checkpoint = False
    start_time = time.time()

    if last_checkpoint_path.exists():
        resume_checkpoint = torch_load(last_checkpoint_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            move_optimizer_state_to_device(optimizer, device)
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        history = list(resume_checkpoint.get("history", []))
        best_metric = float(resume_checkpoint.get("best_val_macro_f1", best_metric))
        best_epoch = resume_checkpoint.get("best_epoch", best_epoch)
        completed_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = completed_epoch + 1
        resumed_from_checkpoint = True
        print(
            f"Resuming {trial_slug} from epoch {completed_epoch}; "
            f"next epoch is {start_epoch}/{config['epochs']}."
        )

    if start_epoch > config["epochs"]:
        print(f"{trial_slug} already has a complete last checkpoint at epoch {start_epoch - 1}.")

    for epoch in range(start_epoch, config["epochs"] + 1):
        print(f"{trial_slug} - Epoch {epoch}/{config['epochs']}")
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate(model, val_loader, criterion, desc="Validate")
        scheduler.step()
        lr_values = [group["lr"] for group in optimizer.param_groups]

        row = {
            "trial_name": trial_slug,
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr_min": min(lr_values),
            "lr_max": max(lr_values),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(trial_dir / "training_history.csv", index=False)
        print(row)

        is_best = val_metrics["macro_f1"] >= best_metric
        if is_best:
            best_metric = float(val_metrics["macro_f1"])
            best_epoch = epoch
            save_json(val_metrics["report"], trial_dir / "best_validation_classification_report.json")
            pd.DataFrame(
                val_metrics["confusion_matrix"],
                index=label_names,
                columns=label_names,
            ).to_csv(trial_dir / "best_validation_confusion_matrix.csv")

        if SAVE_CHECKPOINTS:
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "trial_config": trial_config,
                "label_names": label_names,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_metric,
                "current_val_loss": val_metrics["loss"],
                "current_val_accuracy": val_metrics["accuracy"],
                "current_val_macro_f1": val_metrics["macro_f1"],
                "history": history,
                "model_architecture": MODEL_ARCHITECTURE,
                "text_encoder": TEXT_ENCODER_NAME,
                "encoder_spec": TEXT_ENCODER_SPEC,
                "fusion_block": FUSION_BLOCK_NAME,
                "image_pipeline": IMAGE_PIPELINE_NAME,
                "image_pipeline_source": IMAGE_PIPELINE_SOURCE,
                "checkpoint_policy": CHECKPOINT_POLICY,
            }
            torch.save(checkpoint_payload, last_checkpoint_path)
            if is_best:
                torch.save(checkpoint_payload, best_checkpoint_path)

    if not history:
        raise RuntimeError(f"No training history found for {trial_slug}. Run at least one epoch before summarizing.")

    evaluation_checkpoint_path = best_checkpoint_path if best_checkpoint_path.exists() else last_checkpoint_path
    if evaluation_checkpoint_path.exists():
        evaluation_checkpoint = torch_load(evaluation_checkpoint_path, map_location=device)
        model.load_state_dict(evaluation_checkpoint["model_state_dict"])
    else:
        raise FileNotFoundError(
            f"Missing checkpoint for {trial_slug}. Set SAVE_CHECKPOINTS=True and rerun the trial."
        )

    test_metrics = None
    if RUN_TEST_FOR_EACH_TRIAL:
        test_metrics = evaluate(model, test_loader, criterion, desc="Test")
        save_json(test_metrics["report"], trial_dir / "test_classification_report.json")
        pd.DataFrame(
            test_metrics["confusion_matrix"],
            index=label_names,
            columns=label_names,
        ).to_csv(trial_dir / "test_confusion_matrix.csv")

    elapsed_minutes = (time.time() - start_time) / 60
    summary = {
        **trial_config,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_metric,
        "best_val_accuracy": max(row["val_accuracy"] for row in history),
        "elapsed_minutes": elapsed_minutes,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "resume_start_epoch": start_epoch,
    }
    if test_metrics is not None:
        summary.update(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_weighted_f1": test_metrics["weighted_f1"],
            }
        )
    save_json(summary, trial_dir / "run_summary.json")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary
"""

nb.cells[60].source = """summary_path = OUTPUT_DIR / "trial_summary.csv"
if not summary_path.exists():
    raise FileNotFoundError("Run the trial cells before selecting the best model.")

summary_df = (
    pd.read_csv(summary_path)
    .sort_values(SELECTION_METRIC, ascending=False)
    .reset_index(drop=True)
)

best_trial = summary_df.iloc[0]
best_trial_name = best_trial["trial_name"]
best_trial_dir = OUTPUT_DIR / best_trial_name
best_checkpoint = best_trial_dir / "best_checkpoint.pt"
last_checkpoint = best_trial_dir / "last_checkpoint.pt"

print("Best trial:", best_trial_name)
print("Best checkpoint:", best_checkpoint)
print("Last checkpoint:", last_checkpoint)

display(summary_df)

if not best_checkpoint.exists():
    raise FileNotFoundError(
        f"Missing {best_checkpoint}. Check whether the selected trial finished at least one epoch."
    )
"""

nb.cells[71].source = """---

# **Part 9: Conclusion**

Notebook 05 tunes the selected Bio_ClinicalBERT residual gated fusion model using the original full-image pipeline carried forward from Notebook 04. Notebook 06 remains the final Grad-CAM interpretation step, so the tuning decision made here determines which checkpoint should be explained at the case level.

Each trial saves `last_checkpoint.pt` after every epoch and updates `best_checkpoint.pt` whenever validation macro F1 improves. If a trial stops partway through, rerunning that trial group resumes from the next unfinished epoch instead of starting over.

After rerunning this notebook, select the trial with the strongest validation macro F1, then confirm it against the held-out test metrics and the Notebook 04 original full-image gated-fusion baseline. If no tuning trial improves the validation and test balance, keep the Notebook 04 checkpoint as the stronger final configuration.

---
"""

for cell in nb.cells[6:]:
    if cell.cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

nbformat.write(nb, NOTEBOOK_PATH)
print(f"Added resumable checkpoints to {NOTEBOOK_PATH}")
