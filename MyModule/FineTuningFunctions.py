import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM

class MyFineTunedBert():
    
    def __init__(self):
        pass
    
    def save_model(self, model, path):
        torch.save(model.state_dict(), path)
    
    def load_model(self, path, original_model):
        
        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(original_model)

        # Load the saved state_dict of the model
        state_dict = torch.load(path)

        # Instantiate a new model using the same configuration as the original model
        self.model = BertForMaskedLM.from_pretrained(original_model, state_dict=state_dict)
    
        
    def encode(self, input_sentence):
        
        # Set the model to evaluation mode
        self.model.eval()
        
        if type(input_sentence) != list: input_sentence = [input_sentence]
        
        results = []
        
        for sentence in input_sentence:
        
            # Tokenize a sentence
            tokens = self.tokenizer.tokenize(sentence)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = self.tokenizer.build_inputs_with_special_tokens(ids)

            # Convert the input to a PyTorch tensor
            input_tensor = torch.tensor([input_ids])

            # Generate the model's output
            with torch.no_grad():
                output = self.model(input_tensor)

            # Extract the hidden states from the output
            hidden_states = output[0]

            # Extract the first token's hidden representation as the sentence embedding
            sentence_embedding = hidden_states[0][0]
            
            results.append(sentence_embedding.numpy())
            
        return np.stack(results)