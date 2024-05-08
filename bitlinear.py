# attempt to replace the linear layer in a pretrained model with a bitlinear layer from scratch


from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from bitnet import replace_linears_in_hf

#general preliminaries 
train_data = load_dataset("GATE-engine/mini_imagenet", split='train', trust_remote_code=True).select(range(2000))
val_data = load_dataset("GATE-engine/mini_imagenet", split='validation', trust_remote_code=True).select(range(2000))
test_data = load_dataset("GATE-engine/mini_imagenet", split='test', trust_remote_code=True).select(range(2000))

def get_transform(feature_extractor):
    def transform(examples):
        inputs = feature_extractor(examples["image"], return_tensors="pt")
        # Include the labels
        inputs['labels'] = examples['label']
        return inputs
    return transform

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model_path = "facebook/deit-tiny-patch16-224"
model_name = model_path.split('/')[-1]

feature_extractor = AutoImageProcessor.from_pretrained(model_path)
transform = get_transform(feature_extractor)
model = AutoModelForImageClassification.from_pretrained(model_path).train().to(device)
replace_linears_in_hf(model)

 #general training 
output_dir = "./results/" + model_name
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
    load_best_model_at_end=True,
    dataloader_num_workers=0,
)

base_learning_rate = 1e-3
total_train_batch_size = (
    training_args.train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
)

training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
print("Set learning rate to:", training_args.learning_rate)

#initial training and eval 
print("\n\nINITIAL TRAINING AND EVAL\n\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Load model configuration
config = AutoConfig.from_pretrained("facebook/deit-tiny-patch16-224")

# Define your custom model with the same architecture
class CustomDeiT(nn.Module):
    def __init__(self, config):
        super(CustomDeiT, self).__init__()
        self.model = AutoModelForImageClassification.from_config(config)

    def forward(self, x):
        return self.model(x)

# Instantiate your custom model
model = CustomDeiT(config)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Define Trainer arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=lambda pred: {"accuracy": (pred.predictions.argmax(axis=1) == pred.label_ids).mean()},
)

# Train the model
trainer.train()
metrics = trainer.evaluate(test_dataset.select(range(400)))
trainer.log_metrics("eval", metrics)


# Define data loaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# Instantiate ViT model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        images = batch['pixel_values']
        labels = batch['label']
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['pixel_values']
            labels = batch['label']
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss/len(val_loader):.4f}, Accuracy: {(correct/total)*100:.2f}%')
