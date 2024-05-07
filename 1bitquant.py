
from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from PTQutils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX = 127
MIN = -128

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=False)
        self.fc1 = nn.Linear(16 * 5 * 5, 120, bias=False)
        self.fc2 = nn.Linear(120, 84, bias=False)
        self.fc3 = nn.Linear(84, 10, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

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
    # model =  Net().to(device)  
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
    print("\n\nINITIAL TRAINING AND EVAL\n\n")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()  
    metrics = trainer.evaluate(val_dataset.select(range(400)))
    trainer.log_metrics("eval", metrics)

    visualize_weights(model, './results/before.png')

    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=4,
                                         shuffle=False, num_workers=2)
    
    def test(model: nn.Module, dataloader: DataLoader, max_samples=None) -> float:
        correct = 0
        total = 0
        n_inferences = 0

        with torch.no_grad():
            for data in dataloader:
                images, labels = data['pixel_values'], data['labels']
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if max_samples:
                    n_inferences += images.shape[0]
                    if n_inferences > max_samples:
                        break

        return 100 * correct / total
    score = test(model, testloader)
    print('Accuracy of the network on the test images: {}%'.format(score))

    #quantizing weights and testing
    print('\n\nWEIGHTS HAVE BEEN QUANTIZED, TESTING\n\n')
    quantize_layer_weights(MAX, MIN, model, device)
    visualize_weights(model, './results/after.png')
    for name, layer in list(model.named_modules()):
        if hasattr(layer, 'weight'):
            print(name)
            print(layer.weight.data)
    model2 = model
    trainer = Trainer(
        model=model2,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate(val_dataset.select(range(400)))
    trainer.log_metrics("eval", metrics)
    

    score = test(model2, testloader)
    print('Accuracy of the network on the test images: {}%'.format(score))

    # #profiling
    # print("\n\nREGISTERED HOOKS, PASSING THROUGH FOR CALIBRATION\n\n")
    # register_activation_profiling_hooks(model)
    # metrics = trainer.evaluate(val_dataset.select(range(400)))
    # model.profile_activations = False
    # clear_activations(model)
    # trainer.log_metrics("eval", metrics)


    
    # #quantizing activations and biases
    # print('\n\nQUANTIZING ACTIVATIONS, TESTING\n\n')
    # model3 = modelQuantized(MAX,MIN,model2)
    # metric = evaluate.load("accuracy")

    # trainer = Trainer(
    #     model=model3,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     compute_metrics=compute_metrics,
    # )
    # metrics = trainer.evaluate(val_dataset.select(range(400)))
    # trainer.log_metrics("eval", metrics)


if __name__ == '__main__':
    main()