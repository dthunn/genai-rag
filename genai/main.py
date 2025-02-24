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
print(text_generated)


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

# Preview
print(tokenized_data[:2])


# Isolate the input IDs and the attention masks
inputs_ids = [item['input_ids'].squeeze() for item in tokenized_data]
attention_masks = [item['attention_mask'].squeeze() for item in tokenized_data]
print(attention_masks[:2])


# Convert the input IDs and attention masks to tensors
# This step is necessary for processing the tuned model
inputs_ids = torch.stack(inputs_ids)
attention_masks = torch.stack(attention_masks)