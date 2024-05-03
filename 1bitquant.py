
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

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

train_dataset = load_dataset("imagenet-1k", split='train', trust_remote_code=True).filter(lambda example, idx: idx % 20 == 0, with_indices=True)
test_dataset = load_dataset("imagenet-1k", split='test', trust_remote_code=True).filter(lambda example, idx: idx % 20 == 0, with_indices=True)

model_path = "facebook/deit-tiny-patch16-224"
model_name = model_path.split('/')[-1]


#for deit model first 
feature_extractor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path).train().to(device)


output_dir = "./results/" + model_name

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
trainer.log_metrics("eval", metrics)


score = test(net_with_bias_with_quantized_weights, testloader)

#how to PTQquantise"
''' (TEST) register profiling hoooks, run a testing sweep, pass to quantize layer weights,(TEST) and then netquantized (TEST)'''

register_activation_profiling_hooks(net_with_bias)
test(net_with_bias, trainloader, max_samples=400)
net_with_bias.profile_activations = False
net_with_bias_with_quantized_weights = copy_model(net_with_bias)
quantize_layer_weights(net_with_bias_with_quantized_weights)
score = test(net_with_bias_with_quantized_weights, testloader)

net_quantized_with_bias = NetQuantizedWithBias(net_with_bias_with_quantized_weights)
score = test(net_quantized_with_bias, testloader)