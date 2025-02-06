### SETUP

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

import matplotlib.pyplot as plt
import numpy as np

import os

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output

from IPython.display import display, clear_output

import random
import time
import copy

try:
    import google.colab
    running_on_colab = True
except ImportError:
    running_on_colab = False

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

from datasets import load_dataset

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
import numpy as np

def display_load_dataset():
    global imdb, train_texts, train_labels, test_texts, test_labels

    # Load IMDB dataset
    imdb = load_dataset("imdb")

    # Extract training data
    train_texts = [sample["text"] for sample in imdb["train"]]
    train_labels = torch.tensor([sample["label"] for sample in imdb["train"]])

    # Extract test data
    test_texts = [sample["text"] for sample in imdb["test"]]
    test_labels = torch.tensor([sample["label"] for sample in imdb["test"]])

    global train_size, val_size, imdb_train, imdb_val
    global vectorized_data, vocab_size
    global bow_vectorizer, tfidf_vectorizer

    # Split training set to create a validation set (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(imdb["train"]))
    val_size = len(imdb["train"]) - train_size
    imdb_train, imdb_val = random_split(imdb["train"], [train_size, val_size])

    # Convert back to dataset format
    imdb["train"] = imdb_train
    imdb["validation"] = imdb_val

    vectorized_data = {}
    vocab_size = 5000

    bow_vectorizer = CountVectorizer(max_features=vocab_size, binary=False)
    X_train_bow = bow_vectorizer.fit_transform([ex["text"] for ex in imdb["train"]])
    X_val_bow = bow_vectorizer.transform([ex["text"] for ex in imdb["validation"]])
    X_test_bow = bow_vectorizer.transform(imdb["test"]["text"])
    vectorized_data[f'BoW_{vocab_size}'] = (X_train_bow, X_val_bow, X_test_bow)

    # TF-IDF Model
    tfidf_vectorizer = TfidfVectorizer(max_features=vocab_size)
    X_train_tfidf = tfidf_vectorizer.fit_transform([ex["text"] for ex in imdb["train"]])
    X_val_tfidf = tfidf_vectorizer.transform([ex["text"] for ex in imdb["validation"]])
    X_test_tfidf = tfidf_vectorizer.transform(imdb["test"]["text"])
    vectorized_data[f'TFIDF_{vocab_size}'] = (X_train_tfidf, X_val_tfidf, X_test_tfidf)

    print("Completed Download")

# Example: Get a sample review and label

MAX_DISPLAY_LENGTH = 800

import re

def display_samples(count = 8):
    rows = []
    for _ in range(count):
        sample = random.choice(imdb["train"])  # Cleaner than random index lookup
        # text = re.sub(r"\n\s*\n+", "\n", sample['text']) 
        text = sample['text']
        text = re.sub(r"<br\s*/?>", "<p>", text) 
        label = sample['label']
        rows.append(f"<tr><td style='padding: 10px;'>{'Negative' if label == 0 else 'Positive'}</td><td style='padding: 10px;'>{text[:800]}</td></tr>")

    table_html = f"""
    <table border="1" style="border-collapse: collapse; width: 100%;">
        <tr><th>Label</th><th>Text</th></tr>
        {''.join(rows)}
    </table>
    """
    
    display(widgets.HTML(table_html))


# Define a simple Sequential Neural Network
class SentimentClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, network_input):
        return self.layers(network_input).squeeze(dim=-1)

    def predict_with_sigmoid(self, inputs):
        """
        Perform a forward pass and apply sigmoid to get probabilities.
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.forward(inputs)
        return probabilities
    
    from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

PATIENCE = 5

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    valid_low_loss = float('inf')
    valid_cont_wrong_direction = 0
    valid_best_model_state = copy.deepcopy(model.state_dict())
    valid_best_epoch = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        total_valid_loss = 0
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with torch.no_grad():
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.float())
                total_valid_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs},",
              f"Loss: {total_loss/len(train_loader):.4f},",
              f"Valid Loss: {total_valid_loss/len(val_loader):.4f}")

        if total_valid_loss < valid_low_loss:
            valid_low_loss = total_valid_loss
            valid_cont_wrong_direction = 0
            valid_best_epoch = epoch + 1
            valid_best_model_state = copy.deepcopy(model.state_dict())
        else: 
            valid_cont_wrong_direction += 1
            if valid_cont_wrong_direction >= PATIENCE:
                print("Early Stopping")
                break

    model.load_state_dict(valid_best_model_state)
    print(f"Best Epoch: {valid_best_epoch}")
                
# Function to prepare DataLoader
def prepare_dataloader(X, y, batch_size=32):
    """Convert NumPy arrays or sparse matrices to PyTorch DataLoader."""
    dataset = TensorDataset(torch.tensor(X.toarray(), dtype=torch.float32), 
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def prepare_data():
    global model, train_loader, val_loader
    global X_test_tfidf, y_test

    # Extract train, validation, and test sets
    X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorized_data['TFIDF_5000']
    y_train = [imdb["train"].dataset[i]["label"] for i in imdb["train"].indices]
    y_val = [imdb["validation"].dataset[i]["label"] for i in imdb["validation"].indices]
    y_test = imdb["test"]["label"]  # This remains the same because imdb["test"] is a DatasetDict

    # Convert labels to NumPy arrays for compatibility with TensorDataset
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

    # Create DataLoaders
    train_loader = prepare_dataloader(X_train_tfidf, y_train, batch_size=32)
    val_loader = prepare_dataloader(X_val_tfidf, y_val, batch_size=32)

    # Initialize the model
    input_size = X_train_tfidf.shape[1]
    model = SentimentClassifier(input_size).to(device)

# Train the model
def display_train_model():
    prepare_data()
    train_model(model, train_loader, val_loader, epochs=1)

# Function to evaluate model accuracy
def evaluate_model(model, X_test, y_test, batch_size=32):
    test_loader = prepare_dataloader(X_test, y_test, batch_size)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions = (outputs >= 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    print(f"Test Accuracy: {100 * accuracy:.3f}%")
    return (accuracy, correct, total)

def display_accuracy():
    (accuracy, correct, total) = evaluate_model(model, X_test_tfidf, y_test)
    print(f"Correct: {correct}, Total: {total}")

def get_bow_tfidf(text=None, random_sample=False):
    """
    Converts a given text into BoW and TF-IDF representations, sorted from highest to lowest.
    
    Parameters:
    - text (str, optional): The input text to process. If None and random_sample=True, a random test sample is used.
    - random_sample (bool): If True, selects a random text from the test dataset.
    
    Returns:
    - dict: Containing sorted BoW and TF-IDF representations.
    """
    # Select a random test sample if requested
    if random_sample:
        sample_idx = random.randint(0, len(imdb["test"]["text"]) - 1)
        text = imdb["test"]["text"][sample_idx]
    elif not text:
        raise ValueError("You must provide a text or set random_sample=True.")

    # Convert text to BoW and TF-IDF vectors
    bow_vector = bow_vectorizer.transform([text]).toarray().flatten()
    tfidf_vector = tfidf_vectorizer.transform([text]).toarray().flatten()

    # Sort BoW and TF-IDF values in descending order
    bow_sorted_indices = np.argsort(-bow_vector)
    tfidf_sorted_indices = np.argsort(-tfidf_vector)

    # Extract top words and scores
    bow_words = [(bow_vectorizer.get_feature_names_out()[i], bow_vector[i]) for i in bow_sorted_indices if bow_vector[i] > 0]
    tfidf_words = [(tfidf_vectorizer.get_feature_names_out()[i], tfidf_vector[i]) for i in tfidf_sorted_indices if tfidf_vector[i] > 0]

    return {
        "text": text,
        "BoW": bow_words,
        "TF-IDF": tfidf_words
    }


def display_tfidf_with_prediction(result, model, output):
    """
    Displays TF-IDF results and model prediction in an HTML format inside an Output widget.
    """
    with output:
        output.clear_output()  # Clear previous output before displaying new content

        text = result["text"]
        text = re.sub(r"<br\s*/?>", "<p>", text) 
        bow_words = result["BoW"][:20]  # Show top 20 words
        tfidf_words = result["TF-IDF"][:20]  # Top 20 TF-IDF words

        # Convert text into TF-IDF vector
        tfidf_vector = tfidf_vectorizer.transform([text]).toarray()
        tfidf_tensor = torch.tensor(tfidf_vector, dtype=torch.float32).to(device)  # Convert to tensor

        # Get model prediction (fix: extract scalar value properly)
        probability = model.predict_with_sigmoid(tfidf_tensor).item()  # Extract single value
        predicted_label = "Positive" if probability >= 0.5 else "Negative"
        confidence = f"{max(probability, 1 - probability) * 100:.2f}%"

        # Generate HTML table rows
        rows = []
        for i in range(max(len(bow_words), len(tfidf_words))):
            bow_entry = f"{bow_words[i][0]} ({bow_words[i][1]})" if i < len(bow_words) else ""
            tfidf_entry = f"{tfidf_words[i][0]} ({tfidf_words[i][1]:.4f})" if i < len(tfidf_words) else ""
            rows.append(f"<tr><td style='padding: 8px; width: 150px;'>{bow_entry}</td><td style='padding: 8px; width: 150px;'>{tfidf_entry}</td></tr>")

        # HTML Template
        table_html = f"""
        <div style="font-family: Arial, sans-serif; margin-bottom: 20px;">
            <h3>Original Text:</h3>
            <p style="border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; max-width: 800px; word-wrap: break-word; display: inline-block;">{text}</p>

            <h3>Sentiment Prediction</h3>
            <p><strong>Predicted Sentiment:</strong> {predicted_label}</p>
            <p><strong>Confidence:</strong> {confidence}</p>

            <h3>Top BoW & TF-IDF Words</h3>
            <table border="1" style="border-collapse: collapse; width: 320px; text-align: left;">
                <tr>
                    <th style="padding: 8px; width: 250px; background-color: #f0f0f0;">BoW (Word Count) (Top 20 Only)</th>
                    <th style="padding: 8px; width: 250px; background-color: #f0f0f0;">TF-IDF (Word Score) (Top 20 Only)</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """

        # Display inside Output widget
        display(widgets.HTML(table_html))

random_output_widget = widgets.Output()
random_button = widgets.Button(
    description="Show Random Test Sample",
    button_style="info",  # Optional: 'primary', 'success', 'info', 'warning', 'danger'
    layout=widgets.Layout(width="200px")
)

def on_random_click(_):
    result = get_bow_tfidf(random_sample=True)  # Get random test sample
    display_tfidf_with_prediction(result, model, random_output_widget)  # Pass in the trained model

random_button.on_click(on_random_click)
# Display the output widget (only needs to be done once)

def display_random_sample():
    display(random_button, random_output_widget)
    on_random_click(None)

# Create widgets
eval_text_area = widgets.Textarea(
    placeholder="Enter review here ...",
    layout=widgets.Layout(width='100%')
)

eval_btn = widgets.Button(
    description="Evaulate Review",
    button_style="info",  # Optional: 'primary', 'success', 'info', 'warning', 'danger'
    layout=widgets.Layout(width="200px")
)

eval_output = widgets.Output()

# Define a function to execute when the button is clicked
def on_eval_btn_click(_):
    print("Testing")
    result = get_bow_tfidf(eval_text_area.value)
    display_tfidf_with_prediction(result, model, eval_output)  # Pass in the trained model

eval_btn.on_click(on_eval_btn_click)

# Display widgets
def display_user_written_reviews():
    display(eval_text_area, eval_btn, eval_output)

from ipywidgets import HBox, VBox, HTML, Button, Output
import ipywidgets as widgets

# Output widget and button for incorrect predictions
incorrect_output_widget = Output()
incorrect_button = widgets.Button(
    description="Show More Incorrect Reviews",
    button_style="danger",  # Color to indicate incorrect reviews
    layout=widgets.Layout(width="250px")
)

def get_incorrect_predictions(count=6):
    """
    Retrieve 'count' number of incorrectly labeled reviews with confidence scores.
    """
    incorrect_samples = []
    
    # Run predictions
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(X_test_tfidf.toarray(), dtype=torch.float32).to(device))
        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()  # Get probabilities
        predictions = (probabilities >= 0.5).astype(int)  # Convert to 0 or 1
    
    # Identify misclassified reviews
    incorrect_indices = [i for i in range(len(y_test)) if predictions[i] != y_test[i]]
    random.shuffle(incorrect_indices)  # Shuffle for randomness
    
    for idx in incorrect_indices[:count]:
        text = test_texts[idx]
        true_label = "Positive" if y_test[idx] == 1 else "Negative"
        predicted_label = "Positive" if predictions[idx] == 1 else "Negative"
        confidence = f"{max(probabilities[idx], 1 - probabilities[idx]) * 100:.2f}%"  # Convert to percentage

        incorrect_samples.append(f"""
            <tr>
                <td style='padding: 10px;'>{true_label}</td>
                <td style='padding: 10px;'>
                    {predicted_label} <br> 
                    <span style='font-size: 12px; color: gray;'>Confidence: {confidence}</span>
                </td>
                <td style='padding: 10px;'>{text[:800]}</td>
            </tr>
        """)
    
    return incorrect_samples

def find_incorrect_reviews(_=None):
    """
    Displays incorrectly labeled reviews in an HTML table.
    """
    with incorrect_output_widget:
        incorrect_output_widget.clear_output()
        
        rows = get_incorrect_predictions(6)
        if not rows:
            display(widgets.HTML("<p>No incorrect predictions found.</p>"))
            return
        
        table_html = f"""
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <th>True Label</th>
                <th>Predicted Label</th>
                <th>Review Text</th>
            </tr>
            {''.join(rows)}
        </table>
        """
        
        display(widgets.HTML(table_html))

# Attach event handler to button
incorrect_button.on_click(find_incorrect_reviews)

# Display the button and output widget
def display_incorrect_reviews():
    display(incorrect_button, incorrect_output_widget)
