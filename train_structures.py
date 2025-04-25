import time
from tqdm import tqdm
import torch
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
import torch.nn.functional as F
import pandas as pd
import numpy as np

def train(model, optimizer, criterion, epoch, num_epochs, train_loader, prepare_batch_fn, model_forward_fn):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    device = next(model.parameters()).device  # Get the device from model parameters

    for batch_idx, batch in enumerate(tqdm(train_loader)):
        inputs, labels = prepare_batch_fn(batch)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model_forward_fn(model, *inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = train_loss / (batch_idx + 1)
    epoch_acc = 100. * correct / total
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.3f}, Train Accuracy: {epoch_acc:.3f}')

    return epoch_loss, epoch_acc

def run_training_loop(model, train_loader, optimizer, criterion, num_epochs, 
                      prepare_batch, forward, save_dir):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize tracking variables
    losses = []
    accuracies = []
    csv_path = os.path.join(save_dir, "train_stats.csv")
    model_path = os.path.join(save_dir, "model.pth")

    # Write CSV headers
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Accuracy", "Time (s)"])

    # Training loop
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_loss, epoch_acc = train(
            model, optimizer, criterion, epoch, num_epochs,
            train_loader=train_loader,
            prepare_batch_fn=prepare_batch,
            model_forward_fn=forward
        )
        duration = time.time() - start_time

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, epoch_loss, epoch_acc, duration])

    # Save final model
    torch.save(model.state_dict(), model_path)

    print(f"\n Training complete. Model saved to '{model_path}', stats saved to '{csv_path}'")
    return losses, accuracies

def evaluate(model, test_loader, prepare_batch_fn, forward_fn, inv_act_labels):
    model.eval()
    correct = 0
    top5_correct = 0
    total = 0
    device = next(model.parameters()).device  # Get the device from model parameters
    all_targets = []
    all_predictions = []
    utterance_vectors = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs, targets = prepare_batch_fn(batch)
            inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
            targets = targets.to(device)

            # Extract utterance-level vector from the model
            utt_vecs = model.extract_utterance_vector(*inputs)  # shape: (batch_size, hidden_dim)
            utterance_vectors.extend(utt_vecs.cpu())  # Save as individual tensors for later stacking

            outputs = forward_fn(model, *inputs)

            # Save predictions and targets
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Accuracy metrics
            correct += predicted.eq(targets).sum().item()
            _, top5_pred = outputs.topk(5, 1, True, True)
            target_expanded = targets.view(-1, 1).expand_as(top5_pred)
            top5_correct += top5_pred.eq(target_expanded).sum().item()
            total += targets.size(0)

    acc = 100. * correct / total
    top5_acc = 100. * top5_correct / total
    balanced_acc = 100. * balanced_accuracy_score(all_targets, all_predictions)

    print(f"\nEvaluation Results:")
    print(f"Top-1 Accuracy: {acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc:.2f}%")

    # Convert numeric labels to human-readable labels using inv_act_labels
    all_targets_named = [inv_act_labels[label] for label in all_targets]
    all_predictions_named = [inv_act_labels[label] for label in all_predictions]

    # Classification report and confusion matrix
    print("\n Classification Report:")
    report = classification_report(all_targets_named, all_predictions_named, output_dict=True, zero_division=0)
    print(classification_report(all_targets_named, all_predictions_named, zero_division=0))

    conf_matrix = confusion_matrix(all_targets_named, all_predictions_named)

    return {
        "accuracy": acc,
        "top5_accuracy": top5_acc,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": conf_matrix,
        "classification_report": report,
        "y_true": all_targets_named,
        "y_pred": all_predictions_named,
        "utterance_vectors": utterance_vectors,
    }

def predict_single_input(model, text, speaker_id, target_label=None, is_text_encoded=False,
                         rev_word_map=None, label_map=None, preprocess_fn=None,
                         encode_fn=None, max_len=100):
    """
    Predict a label from a single encoded or raw input.

    Args:
        model: Trained PyTorch model.
        text: Either an encoded tensor (1D) or raw text string.
        speaker_id: Tensor of shape (1, 2), indicating the speaker.
        target_label: Optional, int index of the true label.
        is_text_encoded: Whether the input is already tokenized and tensorized.
        rev_word_map: Mapping from token index to word (optional, for printing).
        label_map: Mapping from label index to label name (optional).
        preprocess_fn: Function to clean the raw text.
        encode_fn: Function to encode cleaned text into token indices.
        max_len: Max length for encoding if raw text is used.

    Returns:
        output: Logits from the model.
        prob: Confidence (probability) of predicted class.
        pred: Index of predicted class.
    """
    print("\nInference Results")
    model.eval()
    device = next(model.parameters()).device  # Get the device from model parameters

    if not is_text_encoded:
        assert preprocess_fn is not None and encode_fn is not None, \
            "Need preprocessing and encoding functions for raw text."

        # Wrap in expected format and preprocess
        wrapped = [{'utterance': text, 'speaker': 'USER' if speaker_id[0][0] == 1 else 'SYSTEM'}]
        _, cleaned = preprocess_fn(wrapped)
        encoded = encode_fn(cleaned, max_len=max_len)
        text_tensor = torch.tensor(encoded)
        
        # Print decoded sentence
        if rev_word_map:
            print("Encoded input:", ' '.join(rev_word_map.get(idx, str(idx)) for idx in encoded[0]))
    else:
        text_tensor = text.unsqueeze(0)  # Shape: (1, seq_len)
        if rev_word_map:
            print("Input:", ' '.join(rev_word_map.get(idx, str(idx)) for idx in text.tolist()))

    text_tensor = text_tensor.to(device)
    speaker_id = speaker_id.to(device)

    with torch.no_grad():
        output = model(text_tensor, speaker_id)
        probs = F.softmax(output, dim=-1)
        prob, pred = probs.max(1)

    # Predicted label
    pred_label = label_map[pred.item()] if label_map else pred.item()
    print(f"Prediction: {pred_label} ({pred.item()}) with probability {prob.item():.3f}")

    # True label
    if target_label is not None:
        true_label = label_map[target_label] if label_map else target_label
        print(f"True Label: {true_label} ({target_label})")

    # Top-5
    print("\nTop-5 Predictions:")
    top5_probs, top5_indices = probs.topk(5, dim=1)
    for idx, p in zip(top5_indices[0], top5_probs[0]):
        label = label_map[idx.item()] if label_map else idx.item()
        print(f"{label}: {p.item():.3f}")

    return output, prob.item(), pred.item()

