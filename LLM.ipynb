import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def handle_non_numeric_values(dataframe, column_name):
    dataframe[column_name] = dataframe[column_name].astype(str).apply(lambda x: x.strip() if x.strip().isdigit() else 'X')

# Load your dataset (assuming it's in a CSV file)
# Replace 'your_dataset.csv' with the actual file name
df = pd.read_csv("drive/MyDrive/Gene/sfari_genes.csv")

# Handle non-numeric values in gene-symbol column
handle_non_numeric_values(df, 'gene-symbol')

# Split the dataset into features (X) and labels (y)
X = df['gene-symbol']
y = df['syndromic']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize and encode the input sequences
train_encodings = tokenizer(list(X_train), truncation=True, padding=True)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True)

# Convert labels to PyTorch tensors
train_labels = torch.tensor(y_train.values,dtype=torch.long)
test_labels = torch.tensor(y_test.values,dtype=torch.long)

# Create PyTorch datasets
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']),
                              torch.tensor(train_encodings['attention_mask']),
                              train_labels)

test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_encodings['attention_mask']),
                             test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load DistilBERT for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
epochs = 3  # You may need to adjust this based on your specific needs

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()

        # Ensure labels are converted to the appropriate dtype
        labels = labels.long()

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.tolist())

# Calculate accuracy and print results
accuracy = accuracy_score(y_test, all_predictions)
classification_report_str = classification_report(y_test, all_predictions)


print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report_str)
