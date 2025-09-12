import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import List, Dict, Union
import os

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class EmailClassificationService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models_dir = "models"
        self.model_path = f"{self.models_dir}/email_classifier"  # Directory to save the model
        self.label_encoder_path = f"{self.models_dir}/label_encoder.joblib"
        self.model_config_path = f"{self.models_dir}/email_classifier/config.json"  # To check if model is trained

    async def train_model(self, dataset_path: str) -> Dict[str, Union[float, int]]:
        """Train the BERT model on the email dataset
        
        Args:
            dataset_path (str): Path to the CSV dataset file
            
        Returns:
            Dict[str, Union[float, int]]: Training results including accuracy and model info
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid or required columns are missing
            RuntimeError: If CUDA is requested but not available
            Exception: For other unexpected errors during training
        """
        try:
            # Validate dataset path
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset file not found at path: {dataset_path}")

            # Create models directory if it doesn't exist
            os.makedirs(self.models_dir, exist_ok=True)

            # Load and preprocess data
            df = pd.read_csv(dataset_path)
            
            # Validate required columns
            required_columns = {'text', 'category'}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns in dataset: {missing_columns}")
            
            # Validate data is not empty
            if df.empty:
                raise ValueError("Dataset is empty")
            
            # Check for null values
            null_counts = df[['text', 'category']].isnull().sum()
            if null_counts.any():
                raise ValueError(f"Dataset contains null values:\n{null_counts}")

            texts = df['text'].values
            labels = df['category'].values

            # Validate unique categories
            unique_categories = df['category'].unique()
            if len(unique_categories) < 2:
                raise ValueError(f"Dataset must contain at least 2 unique categories. Found: {len(unique_categories)}")

            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            try:
                encoded_labels = self.label_encoder.fit_transform(labels)
                joblib.dump(self.label_encoder, self.label_encoder_path)
            except Exception as e:
                raise ValueError(f"Error encoding labels: {str(e)}")

            # Split dataset
            if len(texts) < 10:  # Minimum size for meaningful split
                raise ValueError(f"Dataset too small for training. Found {len(texts)} samples, minimum required: 10")
            
            X_train, X_val, y_train, y_val = train_test_split(
                texts, encoded_labels, test_size=0.2, random_state=42
            )

            # Initialize tokenizer and model for fine-tuning
            try:
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.model = BertForSequenceClassification.from_pretrained(
                    'bert-base-uncased',
                    num_labels=len(self.label_encoder.classes_),
                    problem_type="single_label_classification"
                )
                
                # Ensure the model knows it's being fine-tuned
                self.model.train()
                
                # Initialize weights for the classification head
                # self.model.classifier.apply(self._init_weights)
            except Exception as e:
                raise RuntimeError(f"Error initializing BERT model and tokenizer: {str(e)}")

            # Verify CUDA availability if device is cuda
            if 'cuda' in self.device.type and not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available")
            
            # Move model to device
            self.model.to(self.device)

            # Create datasets and dataloaders
            train_dataset = EmailDataset(X_train, y_train, self.tokenizer)
            val_dataset = EmailDataset(X_val, y_val, self.tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            # Training settings
            optimizer = AdamW(self.model.parameters(), lr=2e-5)
            epochs = 3
            best_accuracy = 0
            training_errors = []

            # Training loop
            for epoch in range(epochs):
                self.model.train()
                total_loss = 0
                batch_count = 0
                
                for batch in train_loader:
                    try:
                        optimizer.zero_grad()
                        
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        loss.backward()
                        optimizer.step()
                        batch_count += 1

                    except Exception as e:
                        training_errors.append(f"Error in batch processing: {str(e)}")
                        continue  # Skip to next batch on error

                # Validation
                val_accuracy = self._validate_model(val_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_accuracy:.4f}")
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    # Save the complete model, tokenizer, and configurations
                    self.model.save_pretrained(self.model_path)
                    self.tokenizer.save_pretrained(self.model_path)

            if training_errors:
                print(f"Warning: encountered {len(training_errors)} errors during training:")
                for err in training_errors[:5]:  # Show first 5 errors
                    print(err)

            if best_accuracy == 0:
                raise RuntimeError("Training failed to produce a valid model (accuracy = 0)")

            return {
                "accuracy": float(best_accuracy),
                "num_classes": len(self.label_encoder.classes_),
                "epochs_trained": epochs,
                "model_path": self.model_path,
                "training_errors": len(training_errors) if training_errors else 0
            }

        except Exception as e:
            raise RuntimeError(f"Training failed: {str(e)}")

    def _init_weights(self, module):
        """Initialize the weights for the classification layer"""
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

        # Training loop
        for epoch in range(epochs):
            try:
                self.model.train()
                total_loss = 0
                batch_count = 0
                
                for batch in train_loader:
                    try:
                        optimizer.zero_grad()
                        
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)

                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )

                        loss = outputs.loss
                        total_loss += loss.item()
                        
                        loss.backward()
                        optimizer.step()
                        batch_count += 1

                    except Exception as e:
                        training_errors.append(f"Error in batch processing: {str(e)}")
                        continue  # Skip to next batch on error

                # Validation
                try:
                    val_accuracy = self._validate_model(val_loader)
                    
                    if val_accuracy > best_accuracy:
                        best_accuracy = val_accuracy
                        # Save the complete model, tokenizer, and configurations
                        self.model.save_pretrained(self.model_path)
                        self.tokenizer.save_pretrained(self.model_path)
                
                except Exception as e:
                    raise RuntimeError(f"Error during validation: {str(e)}")

            except Exception as e:
                raise RuntimeError(f"Error in training epoch {epoch}: {str(e)}")

        if training_errors:
            print(f"Warning: encountered {len(training_errors)} errors during training:")
            for err in training_errors[:5]:  # Show first 5 errors
                print(err)

        if best_accuracy == 0:
            raise RuntimeError("Training failed to produce a valid model (accuracy = 0)")

        return {
            "accuracy": float(best_accuracy),
            "num_classes": len(self.label_encoder.classes_),
            "epochs_trained": epochs,
            "model_path": self.model_path,
            "training_errors": len(training_errors) if training_errors else 0
        }


    def _validate_model(self, val_loader):
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                _, predictions = torch.max(outputs.logits, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_predictions += labels.shape[0]

        return correct_predictions / total_predictions

    async def predict(self, text: str) -> Dict[str, Union[str, float]]:
        """Predict the category of a given text"""
        if self.model is None:
            try:
                # Load the saved model and tokenizer
                self.model = BertForSequenceClassification.from_pretrained(self.model_path)
                self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
                self.label_encoder = joblib.load(self.label_encoder_path)
                self.model.to(self.device)
            except Exception as e:
                raise Exception(f"Failed to load model: Model must be trained first. Error: {str(e)}")

        self.model.eval()
        
        # Tokenize input text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Make prediction
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()

        predicted_category = self.label_encoder.inverse_transform([prediction])[0]

        return {
            "category": predicted_category,
            "confidence": float(confidence)
        }
