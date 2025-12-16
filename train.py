import pandas as pd
import nltk
import re
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# Ensure required NLTK data is downloaded
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
train_df = pd.read_csv('sent_train.csv')
valid_df = pd.read_csv('sent_valid.csv')

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'\$[A-Za-z]+', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.strip()
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def enhanced_preprocess_text(text):
    text = preprocess_text(text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

train_df['text'] = train_df['text'].apply(enhanced_preprocess_text)
valid_df['text'] = valid_df['text'].apply(enhanced_preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df['text']).toarray()
X_valid = vectorizer.transform(valid_df['text']).toarray()
y_train = train_df['label'].values
y_valid = valid_df['label'].values

# Convert to PyTorch dataset
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SentimentDataset(X_train, y_train)
valid_dataset = SentimentDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Define LSTM Model
class LSTMSentiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMSentiment, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(1))
        output = self.fc(self.dropout(lstm_out[:, -1, :]))
        return output

# Model Training
input_dim = X_train.shape[1]
hidden_dim = 256
output_dim = 3
num_layers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMSentiment(input_dim, hidden_dim, output_dim, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}")

# Save model
torch.save(model.state_dict(), 'lstm_sentiment_model.pth')

print("Model trained and saved successfully!")

