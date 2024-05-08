# HPML1bitquantization
Columbia University High Performance Machine Learning project: 1.58 bit quantization for transformers. 

## Project Description
Exploring 1.58 bit optimization in Vision Transformers through different quantization methods
* Post training quantization - weights and activations are quantized after the model has been trained, only impacts inference
* Quantization aware training - training the model with quantization constraints from the beginning
Study and compare quantization techniques in terms of model performance, throughput and model storage

## Repo Outline
* PTQUtils -
* 1bitquant.py - 
* qat.py - 
* bitlinear.py - 

## Commands to Execute

## Results

![Accuracies from Post Training Quantization](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/pqt.png)

![alt text](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/qat.png)

![alt text](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/storage.png)




