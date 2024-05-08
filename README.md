# HPML1bitquantization
Columbia University High Performance Machine Learning project: 1.58 bit quantization for transformers. 

## Project Description
Exploring 1.58 bit optimization in Vision Transformers through different quantization methods
* Post training quantization - weights and activations are quantized after the model has been trained, only impacts inference
* Quantization aware training - training the model with quantization constraints from the beginning
Study and compare quantization techniques in terms of model performance, throughput and model storage.

Project Goals: 
* Explore the various ways one bit quantization can be implemented in Vision models through post training and quantization aware training methods
* Create a Generalized framework to evaluate models through both methods of quantization
* Observe the performance of  models and conclude if Vision models could benefit from 1 bit quantization

## Repo Outline
* PTQUtils - helper functions for Post-training Quantization
* PTQ.py - main file to run Post-training Quantization with different CL options
* quanteval.py - wrapper class that executes Post-training Quantization workflow, called by PTQ.py
* 1bitquant.py - Deprecated file that implements Post Quantization Training methods 
* qat.py - Uses BitNet Architecture to implement Quantization Aware Training - trains models from scratch
* bitlinear.py - Attempt to implementation of BitNet Archinecture

## Commands to Execute
* to run post training quantization - ``` python3 PTQ.py ``` various argparse options as defined in the file, including hf model_path, initial_train, etc. Description can be accessed using argparse help method.
* quantization aware training which trains models from scratch - ``` python3 qat.py ```

## Results and Observations

Accuracies from Post Training Quantization. 

![Accuracies from Post Training Quantization](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/pqt.png)

The accuracy of pretrained models before performing any kind of quantization was ~30%
Assuming this is due to the covariate shifted nature of the Mini ImageNet dataset, we evaluate the quantized models using these scores as baseline. We are concerned only with the difference between the scores of quantized and non-quantized models, and not the absolute scores themselves.
Implementation of PTQ shows poor results, dropping by around 25%. We observe that quantizing weights and activations between [-1,0,1] leads to model scores dropping drastically.

**************************************************************

Metrics from Quantization Aware Training

![alt text](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/qat.png)

While the model performance does end up decreasing with QAT, we see that it retains the performance of the model much better than PQT and only drops by around 5%.
The inference speed on the other hand almost is increased by 2x for both models when applying QAT


**************************************************************

Model Storage with Quantization

![alt text](https://github.com/ramanodgers/HPML1bitquantization/blob/main/docs/resultsImages/storage.png)

There is currently no datatype in native pytorch that can store below int8 precision. The datatype torch.quint4x2 is physically stored as int8 
int2 or 1.58 storage would  require updated hardware and software. 


**************************************************************





