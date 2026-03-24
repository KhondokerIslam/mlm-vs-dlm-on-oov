""" Root file to run the experiment """


from src.dataset_loader import dataset_loader
from src.train import train
from src.test import test

from src.module import discrete
from src.module import nano

import torch
import argparse

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Mask vs Diffusion in Noisy Text Classification")
    
    parser.add_argument(
        "--model_type", 
        type=str,
        default="mlm",
        help="Model Name to use (mlm vs dlm)"
    )

    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="dataset/",
        help="Path to the Train & Val"
    )
    
    parser.add_argument(
        "--test_set_type", 
        type=str, 
        default="base",
        help="Path to the dataset (base/processed)"
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=128,
        help="Number of examples to process in each batch"
    )

    parser.add_argument(
        "--lr", 
        type=int, 
        default=0.01,
        help="Learning Rate"
    )

    parser.add_argument(
        "--epoch", 
        type=int, 
        default=1,
        help="Epoch"
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Utilizing CLI to create new files
    args.test_set_path = f"dataset/test/{args.test_set_type}.tsv"
    args.output_file = f"outputs/{args.model_type}/{args.test_set_type}_result.tsv"

    # Dataloader and Tokenization
    train_loader, val_loader, test_loader = dataset_loader( args.dataset_path, args.test_set_path, args.batch_size, max_length = 64 )

    # Device mapping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu" if args.model_type == "mlm" else device
    device = "cpu"
    print( f"[Notification] Detected Device: {device}" )

    # Defining Model
    model = nano.Classifier( num_labels = 2 ) if args.model_type == "mlm" else discrete.Classifier( num_labels = 2 )
    print( "[Done] Model Defined!" )
    
    # Model Train & Test
    model = train( train_loader, val_loader, model, args.lr, args.epoch, args.model_type, device )
    test( test_loader, args.test_set_path, model, args.output_file, args.model_type, device )


if __name__ == "__main__":
    main()