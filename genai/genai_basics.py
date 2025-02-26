from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# Initialize the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")



def simple_text_generation(prompt, model, tokenizer, max_length = 100):
  # Encoding the prompt to get the input ids
  input_ids = tokenizer.encode(prompt, return_tensors="pt") # pt = pytorch

  # Generate text using the model
  outputs = model.generate(input_ids, max_length = 100)

  # Decode the generated output IDs back into text
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test the function
prompt = "Dear boss ... "
text_generated = simple_text_generation(prompt,
                                        model,
                                        tokenizer,
                                        max_length = 100)


data = [
    "This paper presents a new method for improving the performance of machine learning models by using data augmentation techniques.",
    "We propose a novel approach to natural language processing that leverages the power of transformers and attention mechanisms.",
    "In this study, we investigate the impact of deep learning algorithms on the accuracy of image recognition tasks.",
    "Our research demonstrates the effectiveness of transfer learning in enhancing the capabilities of neural networks.",
    "This work explores the use of reinforcement learning for optimizing decision-making processes in complex environments.",
    "We introduce a framework for unsupervised learning that significantly reduces the need for labeled data.",
    "The results of our experiments show that ensemble methods can substantially boost model performance.",
    "We analyze the scalability of various machine learning algorithms when applied to large datasets.",
    "Our findings suggest that hyperparameter tuning is crucial for achieving optimal results in machine learning applications.",
    "This research highlights the importance of feature engineering in the context of predictive modeling."
]


# Tokenization
# All inputs must have the same length
# Ensure all inputs have the same length by adding a dummy token at the end
# This process of adding dummy tokens is called padding.
tokenizer.pad_token = tokenizer.eos_token


# Tokenize the data
tokenized_data = [tokenizer.encode_plus(
    sentence,                   # Input sentence
    add_special_tokens = True,  # Add special tokens
    return_tensors = "pt",      # Return PyTorch tensors
    padding = "max_length",     # Pad to the maximum length
    max_length = 50)            # Maximum length of the sequence
    for sentence in data]       # Iterate over the data


# Isolate the input IDs and the attention masks
inputs_ids = [item['input_ids'].squeeze() for item in tokenized_data]
attention_masks = [item['attention_mask'].squeeze() for item in tokenized_data]


# Convert the input IDs and attention masks to tensors
# This step is necessary for processing the tuned model
inputs_ids = torch.stack(inputs_ids)
attention_masks = torch.stack(attention_masks)


# Padding all input sequences to ensure they have the same length
padded_input_ids = pad_sequence(
    inputs_ids,
    batch_first = True,
    padding_value = tokenizer.eos_token_id) # Use the tokenizer's end-of-sequence token ID as the padding value


# Padding all attention masks to ensure they have the same length
padded_attention_masks = pad_sequence(
    attention_masks,
    batch_first = True,
    padding_value = 0) # Use 0 as the padding value for attention masks


# Create a custom dataset class for handling text data, including input IDs and attention masks
class TextDataset(Dataset):
  def __init__(self, input_ids, attention_masks):
    self.input_ids = input_ids  # Store the input IDs
    self.attention_masks = attention_masks  # Store the attention masks
    self.labels = input_ids.clone()  # Create labels identical to input IDs for tasks like language modeling

  def __len__(self):
    return len(self.input_ids)  # Return the number of samples in the dataset

  def __getitem__(self, idx):
    # Return a dictionary containing the input IDs, attention mask, and labels for a given index
    return {
        'input_ids': self.input_ids[idx],
        'attention_mask': self.attention_masks[idx],
        'labels': self.labels[idx]
    }


# Instantiate the dataset using the padded input IDs and attention masks
dataset = TextDataset(padded_input_ids, padded_attention_masks)


# Prepare the data in batches using a DataLoader
# Set the batch size to 2 and shuffle the data for each epoch
dataloader = DataLoader(dataset, batch_size = 2, shuffle = True)


# Initialize an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5)


# Set the model to training mode
model.train()


# Training loop
for epoch in range(10):
  for batch in dataloader:
    # Unpacking the input and attention mask ids
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # Reset the gradients to zero
    optimizer.zero_grad()

    # Forward pass
    # Processing the input and attention masks
    outputs = model(input_ids = input_ids,
                    attention_mask = attention_mask,
                    labels = input_ids)
    loss = outputs.loss

    # Backward pass: compute the gradients of the loss
    loss.backward()

    # Update the model parameters
    optimizer.step()

  # Print the loss for the current epoch to monitor the progress
  print(f"Epoch {epoch + 1} - Loss: {loss.item()}")


  # Define a function to generate text given a prompt
def generate_text(prompt, model, tokenizer, max_length=100):
  # Encode the prompt to obtain input IDs and attention mask
  inputs = tokenizer.encode_plus(prompt, return_tensors="pt")
  # Extract input ids and attention mask
  input_ids = inputs['input_ids']
  attention_mask = inputs['attention_mask']

  # Generate text using the model
  outputs = model.generate(
      input_ids,                       # Provide input IDs to the model
      attention_mask=attention_mask,   # Provide attention mask to the model
      max_length=max_length            # Set the maximum length for the generated text
  )

  # Decode the generated text and return it, skipping special tokens
  return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Test the function
prompt = "In this research, we "
text_generated = generate_text(prompt, model, tokenizer, max_length = 500)
print(text_generated)


# Test the function
prompt = "Dear Boss ..."
text_generated = generate_text(prompt, model, tokenizer, max_length = 500)
print(text_generated)