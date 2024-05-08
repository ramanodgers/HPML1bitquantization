# Quantization Aware Training (QAT) for Vision Transformers from scratch


from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoConfig
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

from PTQutils import *

from bitnet import replace_linears_in_hf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def main():

    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        # Access command-line arguments
        model_arg = sys.argv[1]
        num_epochs_arg = sys.argv[2] if len(sys.argv) > 2 else None

        # Print the command-line arguments
        print("Model being trained: ", model_arg)
        print("Training for: " + num_epochs_arg + " epochs")
    else:
        model_path = "facebook/deit-tiny-patch16-224"
        print("No command-line arguments provided., using default model: ", model_path)

    # loading model architecture
    if model_arg == "vit":
        model_path = 'google/vit-base-patch16-224-in21k'
    elif model_arg == "deit":
        model_path = "facebook/deit-tiny-patch16-224"


    model_name = model_path.split('/')[-1]

    # loading model and feature extractor
    feature_extractor = AutoImageProcessor.from_pretrained(model_path)
    transform = get_transform(feature_extractor)
    model = AutoModelForImageClassification.from_pretrained(model_path).train().to(device)

    # model =  Net().to(device)  
    output_dir = "./results/" + model_name

    # loading dataset
    train_dataset = train_data.with_transform(transform)
    val_dataset = val_data.with_transform(transform)
    test_dataset = test_data.with_transform(transform)

    #general training 
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_epochs_arg if num_epochs_arg else 1,
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
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()  
    metrics = trainer.evaluate(test_dataset.select(range(400)))
    trainer.log_metrics("eval", metrics)

    visualize_weights(model, './results/before.png')

    print("AFTER QUANTIZATION...")

    # Quantizating the model
    replace_linears_in_hf(model)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()  
    metrics = trainer.evaluate(test_dataset.select(range(400)))
    trainer.log_metrics("eval", metrics)

if __name__ == '__main__':
    main()