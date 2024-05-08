
import argparse
from transformers import TrainingArguments, Trainer
from quanteval import QuantEval
import torch 
import os


parser = argparse.ArgumentParser(description="Quantization of Neural Network Models")
parser.add_argument('--model_path', type=str, default='facebook/deit-tiny-patch16-224', 
                    help='what model to quantize')
parser.add_argument('--imagenet', type=bool, default=True, 
                    help='whether to use ImageNet or CIFAR')
parser.add_argument('--freeze', type=bool, default=False, 
                    help='whether to freeze model in training')
parser.add_argument('--initial_train', type=bool, default=True, 
                    help='whether to train model before quantizing')
parser.add_argument('--quantize_weight', type=bool, default=True, 
                    help='whether to quantize weights')
parser.add_argument('--quantize_ab', type=bool, default=True, 
                    help='whether to quantize activations and biases')
parser.add_argument('--range', type=str, default='ternary', choices=['ternary', 'int8'],
                    help='which quantization to use')

workflow = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MAX = 127
# MIN = -128

def main():
    #model specific 
    # cifar = False
    # model_path = "facebook/deit-tiny-patch16-224"
    # model_path = "google/vit-base-patch16-224"

    # model_name = model_path.split('/')[-1]
    # output_dir = "./results/" + model_name

    #config, trainset, freeze
    model_name = workflow.model_path.split('/')[-1]
    output_dir = os.path.join("./results/", model_name, workflow.range) + '/'
    
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
    runner = QuantEval(training_args, device, workflow)
    out_model = runner.conduct_workflow()

if __name__ == '__main__':
    main()