
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from tqdm import tqdm

from PTQutils import *

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
    #model specific 
    model_path = "facebook/deit-tiny-patch16-224"
    model_name = model_path.split('/')[-1]

    feature_extractor = AutoImageProcessor.from_pretrained(model_path)
    transform = get_transform(feature_extractor)
    model = AutoModelForImageClassification.from_pretrained(model_path).train().to(device)
    output_dir = "./results/" + model_name

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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train() 

    #profiling
    register_activation_profiling_hooks(model)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    #quantizing weights and testing
    quantize_layer_weights(model)
    model2 = model
    trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()

    #quantizing activations and biases
    model3 = NetQuantized(model2)
    trainer = Trainer(
        model=model3,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()


    # score = test(net_with_bias_with_quantized_weights, testloader)

    # #how to PTQquantise"
    # ''' (TEST) register profiling hoooks, run a testing sweep, pass to quantize layer weights,(TEST) and then netquantized (TEST)'''

    # register_activation_profiling_hooks(net_with_bias)
    # test(net_with_bias, trainloader, max_samples=400)
    # net_with_bias.profile_activations = False
    # net_with_bias_with_quantized_weights = copy_model(net_with_bias)
    # quantize_layer_weights(net_with_bias_with_quantized_weights)
    # score = test(net_with_bias_with_quantized_weights, testloader)

    # net_quantized_with_bias = NetQuantizedWithBias(net_with_bias_with_quantized_weights)
    # score = test(net_quantized_with_bias, testloader)

if __name__ == '__main__':
    main()