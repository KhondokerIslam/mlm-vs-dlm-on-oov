""" Test Structure Defining File """

import pandas as pd
import math

import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def loader( file_path ):

    data = pd.read_csv(file_path, sep='\t')
    return data

def save_fr_analysis( all_labels, all_preds, test_set_path, output_file ):
    """
        Saving Results in Structured manner for analysis
    """
    
    test_data = loader( file_path = test_set_path )

    test_data['actual_label'] = all_labels
    test_data['predicted_label'] = all_preds

    test_data.to_csv( output_file, sep='\t', index=False)

def sample_sigma(batch_size, sigma_min=1e-4, sigma_max=20.0, device="cuda"):
        return torch.exp(
            torch.rand(batch_size, device=device) *
            (math.log(sigma_max) - math.log(sigma_min)) +
            math.log(sigma_min)
        ) 

def test( test_loader, test_set_path, model, output_file, model_type, device ):
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            if( model_type == "dlm" ):
                batch_size = input_ids.size(0)
                sigma = sample_sigma( batch_size, device=device) 
                logits = model(input_ids, sigma, attention_mask=attention_mask, labels=labels)

            else:
                logits = model(input_ids, attention_mask=attention_mask, labels=labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print( "[Done] Testing Complete!" )
    print(f'Test - Accuracy: {accuracy:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1 Score: {f1:.4f}')

    save_fr_analysis( all_labels, all_preds, test_set_path, output_file )

    print( f"[Done] Analysis Saved to {output_file}!" )