


from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F 

import torchvision

from PTQutils import *
   
def get_transform(feature_extractor, imagenet = True):

    def transform(examples):
        inputs = feature_extractor(examples["image"], return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs
    
    def cifar_transform(examples):
        inputs = feature_extractor(examples["img"], return_tensors="pt")
        inputs['labels'] = examples['label']
        return inputs
    
    if imagenet: 
        return transform
    else:
        return cifar_transform

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
 
 
class QuantEval():
    def __init__(self, training_args, device, workflow):
        self.training_args = training_args
        self.workflow = workflow
        self.device = device
        if self.workflow.range == 'ternary':
            self.max = 1
            self.min = -1
        else:
            self.max = 127
            self.min = -128

        self.feature_extractor = AutoImageProcessor.from_pretrained(self.workflow.model_path)
        self.model = AutoModelForImageClassification.from_pretrained(self.workflow.model_path).to(device)
        self.config = self.model.config
        self.config.num_labels = 1000 if self.workflow.imagenet else 10
        self.model = AutoModelForImageClassification.from_config(self.config).to(device)
        self.set_lr()

        self.transform = get_transform(self.feature_extractor, imagenet = self.workflow.imagenet)
        self.data = QuantEval.get_data(self.transform, self.workflow.imagenet)

    def set_lr(self):
        base_learning_rate = 1e-3
        total_train_batch_size = (
            self.training_args.train_batch_size * self.training_args.gradient_accumulation_steps * self.training_args.world_size
        )

        self.training_args.learning_rate = base_learning_rate * total_train_batch_size / 256
        print("Set learning rate to:", self.training_args.learning_rate)

    def conduct_workflow(self):
        if self.workflow.initial_train:
            self.initial_train()
        if self.workflow.quantize_weight or self.workflow.quantize_ab:
            self.weight_quantize()
        if self.workflow.quantize_ab:
            self.ab_quantize()
        return self.model
        
    def initial_train(self):
        print("\n\nINITIAL TRAINING AND EVAL\n\n")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['test'],
            compute_metrics=compute_metrics,
        )
        if self.workflow.freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        trainer.train()  
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        visualize_weights(self.model, self.training_args.output_dir + 'initial_train.png')

    def weight_quantize(self):
        print('\n\nWEIGHTS HAVE BEEN QUANTIZED, TESTING\n\n')
        quantize_layer_weights(self.max, self.min, self.model, self.device)
        visualize_weights(self.model, self.training_args.output_dir + 'weight_q.png')

        model2 = self.model
        trainer = Trainer(
            model=model2,
            args=self.training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['test'],
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        self.model = model2

    def ab_quantize(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['test'],
            compute_metrics=compute_metrics,
        )
        #profiling
        print("\n\nREGISTERED HOOKS, PASSING THROUGH FOR CALIBRATION\n\n")
        register_activation_profiling_hooks(self.model)
        metrics = trainer.evaluate()
        self.model.profile_activations = False
        clear_activations(self.model)

        #We found that repeated trainers were necessary to avoid nasty hf bugs
        #quantizing activations and biases
        print('\n\nQUANTIZING ACTIVATIONS, TESTING\n\n')
        model3 = modelQuantized(self.max,self.min,self.model)
        metric = evaluate.load("accuracy")

        trainer = Trainer(
            model=model3,
            args=self.training_args,
            train_dataset=self.data['train'],
            eval_dataset=self.data['test'],
            compute_metrics=compute_metrics,
        )
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        self.model = model3
    
    @staticmethod
    def get_data(transform, imagenet = True):
        if imagenet:
            train_data = load_dataset("GATE-engine/mini_imagenet", split='train', trust_remote_code=True).select(range(2000))
            test_data = load_dataset("GATE-engine/mini_imagenet", split='test', trust_remote_code=True).select(range(2000))
        else:
            train_data = load_dataset("cifar10", split='train', trust_remote_code=True).select(range(10000))
            test_data = load_dataset("cifar10", split='test', trust_remote_code=True).select(range(2000))
        
        train_dataset = train_data.with_transform(transform)
        test_dataset = test_data.with_transform(transform)

        return {'train':train_dataset, 'test':test_dataset}



    
    

    


