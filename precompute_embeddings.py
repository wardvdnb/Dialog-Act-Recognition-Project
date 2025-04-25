import os
import torch
import json
import torch.utils.data
from tqdm import tqdm
import numpy as np
import torch
import json
from tqdm import tqdm

def extract_feature_vectors(sentence, embed, encoder, type="last", device='cuda'):
    """
    Extract feature vectors using GPU
    """
    encoder.eval()
    # Move models to GPU
    embed.to(device)
    encoder.to(device)
    
    if not sentence:  # Handle empty utterances
        output_size = encoder.h2o.weight.shape[1]  # Get the embedding size dynamically
        return torch.zeros(output_size).cpu()  # Return a zero vector

    with torch.no_grad():
        # Create tensor on GPU directly
        sentence_tensor = torch.LongTensor([ord(c) for c in sentence]).unsqueeze(1).to(device)  # Shape: (seq_len, 1)
        
        # Get initial state and move to device
        hidden_state = encoder.state0(batch_size=1)
        h_0, c_0 = hidden_state
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        hidden_state = (h_0, c_0)
        
        hidden_states = []  # Store all hidden states
        
        for char in sentence_tensor:
            emb = embed(char)
            hidden_state, _ = encoder(emb, hidden_state)
            # Only move to CPU after processing is complete
            hidden_states.append(hidden_state[0].detach())  # Keep on GPU for now
        
        # Stack all hidden states on GPU and then move to CPU at the end
        stacked_states = torch.stack(hidden_states)
        
        # Last hidden state
        last_feature_vector = stacked_states[-1].squeeze()
        
        # Average hidden state
        avg_feature_vector = torch.mean(stacked_states, dim=0).squeeze()
        
        # Return to CPU only at the end
        if type == "last":
            return last_feature_vector.cpu()
        else:
            return avg_feature_vector.cpu()

def precompute_embeddings(file_path, embed, encoder, output_path, type="average", device='cuda'):
    """
    Precompute embeddings using GPU
    """
    with open(file_path, 'r') as f:
        file_contents = json.load(f)
        dialogues = [dialogue['turns'] for dialogue in file_contents]
        all_embeddings = []
        
        for dialog in tqdm(dialogues, desc="Processing dialogues for file {}".format(output_path)):
            embeddings = [extract_feature_vectors(turn['utterance'], embed, encoder, type=type, device=device) for turn in dialog]
            all_embeddings.append(embeddings)
        
        # Save embeddings to disk
        torch.save(all_embeddings, output_path)

def run_embedding_extraction(model_path='mlstm-ns.pt', path_to_train=None, path_to_test=None):
    
    # First check if the embeddings.pt files already exist
    if os.path.exists('train_embeddings_average.pt') and os.path.exists('test_embeddings_average.pt'):
        print("Embeddings already exist. Skipping extraction.")
        return

    # Check if CUDA is available, otherwise fall back to CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model to specified device
    checkpoint = torch.load(model_path, map_location=device)
    encoder = checkpoint['rnn']
    embed = checkpoint['embed']
    
    # Process train data
    if path_to_train:
        print("Processing training data...")
        precompute_embeddings(path_to_train, embed, encoder, 
                             'train_embeddings_average.pt', 
                             type="average", device=device)
        
        # precompute_embeddings(path_to_train, embed, encoder, 
        #                      'train_embeddings_last.pt', 
        #                      type="last", device=device)
    
    # Process test data
    if path_to_test:
        print("Processing test data...")
        precompute_embeddings(path_to_test, embed, encoder, 
                             'test_embeddings_average.pt', 
                             type="average", device=device)
        
        # precompute_embeddings(path_to_test, embed, encoder, 
        #                      'test_embeddings_last.pt', 
        #                      type="last", device=device)
    
    print("All processing complete!")