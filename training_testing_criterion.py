# Importing Torch Libraries
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, recall_score


# Training Method
def train(model, device, train_queue, valid_queue, optimizer, epoch, criterion):
    # LSTM Model in Train Mode, total loss to 0
    model.train()
    loss_total = 0
    total = 0

    # Iteration over whole training data
    for data, target, l in train_queue:
        # data, target (y value) to GPU
        data = data.to(device)
        target = target.to(device)
        data = data.long()

        # FORWARD PASS
        optimizer.zero_grad()
        output = model(data, l)

        # BCE-Loss with Logits
        loss = criterion(output, target.float())

        # BACKWARD AND OPTIMIZE
        loss.backward()
        optimizer.step()

        # Loss added for all the records
        loss_total += loss.item() * target.shape[0]

        total += target.shape[0]

    # Test called For Validation of LSTM model
    val_loss, val_acc, roc_macro, roc_weighted, recall_macro, precision_weighted = \
        test(model, device, valid_queue, criterion)
    if epoch % 100 == 0:
        print("Epoch: %d, Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, Val AUC_ROC Macro: %.3f, Val AUC_ROC "
              "Weighted : %.3f, Val Recall Macro: %.3f, Val Precision Weighted : %.3f" %
              (epoch + 1, loss_total / total, val_loss, val_acc, roc_macro, roc_weighted, recall_macro,
               precision_weighted))


# Validation Method
def test(model, device, valid_queue, criterion):
    # LSTM Model in eval mode , total loss to 0
    model.eval()
    y_true = []
    y_pred = []
    y_scores = []
    loss_total = 0
    total = 0

    # requires_grad flag to false for the test mode
    with torch.no_grad():
        for data, target, l in valid_queue:
            # data, target (y value) to GPU
            data, target = data.to(device), target.to(device)
            data = data.long()

            # FORWARD PASS
            output = model(data, l)

            # Validation Loss using BCE-Loss with Logits
            loss = criterion(output, target.float())

            # Loss added for all the records
            loss_total += loss.item() * target.shape[0]

            total += target.shape[0]

            # PREDICTIONS
            scores = torch.sigmoid(output)
            pred = torch.round(scores).int()
            target = target.int()
            y_scores.extend(scores.reshape(-1).tolist())
            y_true.extend(target.reshape(-1).tolist())
            y_pred.extend(pred.reshape(-1).tolist())

        # Calculating Accuracy, Recall, Precision ,Area Under Curve Scores for Multi-Label Classification
        return loss_total / total, accuracy_score(y_true, y_pred), roc_auc_score(y_true, y_scores), \
               roc_auc_score(y_true, y_scores, average="weighted"), recall_score(y_true, y_pred, average='macro'), \
               precision_score(y_true, y_pred, average='weighted')


# Method to return Loss Function , Optimizer , scheduler for Optimizer
def get_criterion_optimizer_scheduler(model, num_epochs, lr, loss_weights, device):
    # Loss Function for Multi-label Classification i.e. BCE-Loss with Logits with weights to handle imbalance
    criterion = nn.BCEWithLogitsLoss(pos_weight=loss_weights)
    criterion = criterion.to(device)

    # Stochastic Gradient Descent
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=3e-4
    )

    # Cosine Annealing with SGD to obtain Best Result after optimal number of epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    return criterion, optimizer, scheduler
