import torch
import pandas as pd
import numpy as np
import evaluate
import os
import seaborn as sns
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    TrainingArguments,
    Trainer
)

# --- 1. Configuration & Setup ---

FILE_PATH = r"data\cleaned\deep_learning_data.csv"
MODEL_CHECKPOINT = "distilbert-base-uncased"
RANDOM_STATE = 100

MAX_TOKEN_LENGTH = 512 
NUM_TRAIN_EPOCHS = 20  
TEXT_COLUMN_NAME = "description" 
LABEL_COLUMN_NAME = "urgency"   

# --- CHANGED: Updated for Binary Classification (2 Labels) ---
NUM_LABELS = 2
id2label = {0: "NO", 1: "YES"}
label2id = {"NO": 0, "YES": 1}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Data Loading, Cleaning, Splitting & OVERSAMPLING ---

if not os.path.exists(FILE_PATH):
    print(f"\n❌ ERROR: File not found at {FILE_PATH}. Please check the path or upload the data.")
    exit()

try:
    df = pd.read_csv(FILE_PATH)
    
    # Handle Missing Text Data 
    df = df.dropna(subset=[TEXT_COLUMN_NAME])
    
    # Ensure the label column is an integer 
    if df[LABEL_COLUMN_NAME].dtype != 'int64':
         df[LABEL_COLUMN_NAME] = df[LABEL_COLUMN_NAME].astype(int) 

    # --- SAFETY CHECK: Ensure data only contains 0 and 1 ---
    df = df[df[LABEL_COLUMN_NAME].isin([0, 1])]

    # 2.1. Split data (70% Train, 15% Val, 15% Test)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE, shuffle=True)

    print(f"Original Train size: {len(train_df)}")

    # --- OVERSAMPLING THE TRAINING DATA ---
    print("\n  Applying Oversampling to Training Data...")
    
    # 1. Find the count of the majority class
    class_counts = train_df[LABEL_COLUMN_NAME].value_counts()
    max_count = class_counts.max()
    print(f"   Target count per class: {max_count}")
    
    # 2. Create a list to hold the balanced data
    balanced_dfs = []
    
    # 3. Loop through each class ID
    for label_id in train_df[LABEL_COLUMN_NAME].unique():
        # Get all rows for this specific class
        class_subset = train_df[train_df[LABEL_COLUMN_NAME] == label_id]
        
        # Resample (duplicate) this subset to match the max_count
        oversampled_subset = class_subset.sample(n=max_count, replace=True, random_state=RANDOM_STATE)
        balanced_dfs.append(oversampled_subset)
    
    # 4. Concatenate and Shuffle
    train_df = pd.concat(balanced_dfs)
    train_df = shuffle(train_df, random_state=RANDOM_STATE) 
    
    print(f"   New Balanced Train size: {len(train_df)}")
    print(f"   New Class Distribution:\n{train_df[LABEL_COLUMN_NAME].value_counts()}")
    # ---------------------------------------------------

    print(f"\nFinal Train samples: {train_df.shape[0]}")
    print(f"Validation samples: {val_df.shape[0]}")
    print(f"Test samples: {test_df.shape[0]}")

    # Convert Pandas DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
    eval_dataset = Dataset.from_pandas(val_df, preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

except Exception as e:
    print(f"\n❌ ERROR during data loading/splitting: {e}")
    import traceback
    traceback.print_exc()
    exit()

# --- 3. Tokenizer and Preprocessing ---

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)

def tokenize_function(examples):
    return tokenizer(
        examples[TEXT_COLUMN_NAME], 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_TOKEN_LENGTH
    )

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.rename_column(LABEL_COLUMN_NAME, "labels").remove_columns([TEXT_COLUMN_NAME])
tokenized_eval_dataset = tokenized_eval_dataset.rename_column(LABEL_COLUMN_NAME, "labels").remove_columns([TEXT_COLUMN_NAME])
tokenized_test_dataset = tokenized_test_dataset.rename_column(LABEL_COLUMN_NAME, "labels").remove_columns([TEXT_COLUMN_NAME])


# --- 4. Model Loading and Metric Definition ---

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
).to(device)

metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = metric.compute(predictions=predictions, references=labels)

    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {**acc, **f1}  # type: ignore


# --- 5. Training Arguments and Trainer Setup ---

training_args = TrainingArguments(
    output_dir="./distilbert_urgency_results", 
    num_train_epochs=NUM_TRAIN_EPOCHS,      
    per_device_train_batch_size=16,           
    per_device_eval_batch_size=16,            
    learning_rate=2e-5,                  
    weight_decay=0.01,
    logging_steps=50,
    eval_strategy="epoch",         
    save_strategy="epoch",               
    load_best_model_at_end=True,         
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics,
)

# --- 6. Start Training, Evaluation, and Saving ---

print("\n\n##################################")
print("##### Starting Fine-Tuning #####")
print("##################################")

trainer.train()

print("\nValidation Set Evaluation:")
validation_results = trainer.evaluate()
print(validation_results)

print("\nTest Set Evaluation (Unseen Data):")
test_results = trainer.evaluate(tokenized_test_dataset)
print(test_results)

# 1. Get predictions on the validation set (used for finding the best model)
print("\n\n##################################")
print("##### Running Diagnostics ######")
print("##################################")
print("Generating predictions on the Validation Set...")

predictions_output = trainer.predict(tokenized_eval_dataset) 
preds = np.argmax(predictions_output.predictions, axis=-1)
true_labels = predictions_output.label_ids

# 2. Print the Classification Report
print("\n🔍 CLASSIFICATION REPORT (Validation Set):")

print(classification_report(true_labels, preds, target_names=["NO", "YES"]))

# 3. Save the final best model
SAVE_PATH = "./deep_learning/final_distilbert_urgency_model"
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)
print(f"\nModel and Tokenizer saved to '{SAVE_PATH}'")