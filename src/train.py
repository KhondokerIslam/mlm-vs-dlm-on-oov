import torch
import torch.nn as nn
import math

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Focal loss function for handling class imbalance
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean() if self.reduction == 'mean' else F_loss.sum()

class Train:

    def __init__(self, train_loader, val_loader, model, lr, epoch, model_type, device):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model_type = model_type    
        
        self.lr = lr
        self.epoch = epoch

        total_steps = len(train_loader) * self.epoch

        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=2e-5,
            weight_decay=0.3
        )


        self.total_steps = len(train_loader) * epoch
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        self.model.to(self.device)

        ## loss defining
        self.loss_fn = FocalLoss(alpha=1.0, gamma=2.0).to(self.device)    
    

    def sample_sigma(self, batch_size, sigma_min=1e-4, sigma_max=20.0, device="cuda"):
        return torch.exp(
            torch.rand(batch_size, device=device) *
            (math.log(sigma_max) - math.log(sigma_min)) +
            math.log(sigma_min)
        ) 

    
    def train( self ):
        best_val_loss = 1000.
        no_improvement_epochs = 0
        early_stop_tolerance = 2

        for epoch in range(self.epoch):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                if( self.model_type == "dlm" ):

                    batch_size = input_ids.size(0)

                    ## definating noise sample
                    sigma = self.sample_sigma( batch_size, device=self.device)  # adjust range if needed
                    logits = self.model(input_ids, sigma, attention_mask=attention_mask, labels=labels)

                else:
                    logits = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                loss = self.loss_fn(logits, labels)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item()

            print(f"[Epoch {epoch + 1}] Training Loss: {total_loss / len(self.train_loader)}")

            # Validation phase
            val_accuracy, val_precision, val_f1, val_loss = self.val()
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= early_stop_tolerance:
                    print("Early stopping triggered")
                    break            
    
    def val( self ):
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]

                if( self.model_type == "dlm" ):

                    batch_size = input_ids.size(0)
                    sigma = self.sample_sigma( batch_size, device=self.device)  # adjust range if needed
                    logits = self.model(input_ids, sigma, attention_mask=attention_mask, labels=labels)

                else:
                    logits = self.model(input_ids, attention_mask=attention_mask, labels=labels)

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()

        accuracy = accuracy_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        print(f'Validation - Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f} | Loss: {total_loss / len(self.val_loader)}')

        return accuracy, precision, f1, total_loss / len(self.val_loader)

def train( train_loader, val_loader, model, lr, epoch, model_type, device ):

    train = Train( train_loader, val_loader, model, lr, epoch, model_type, device )
    train.train()

    print( "[Done] Training Complete!" )

    return train.model

    




