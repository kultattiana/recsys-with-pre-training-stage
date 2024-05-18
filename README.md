# recsys-with-pre-training-stage
## Recommendation System with a Pre-Training Stage Based on Neural Networks for Event Sequence Processing

### The idea
This work constructs a deep learning model capable of processing event
sequences and considering users’ connections to predict their
subsequent actions. Also, a pre-training phase that extracts useful
representations of spatial-temporal data about users’ behavior
and social relationships is added.

### The structure
The project is organized as follows:
- `research` folder consists of notebooks with data load and preprocessing scripts, experiments with graph pre-training and rnn baseline, statistical tests and metrics calculations
- `pytorch_lifestream_experiments` folder contains pytorch lifestream models configs, data preprocessing with PySpark scripts, commands for launching models and the notebook with experiments on sequences pre-trained embeddings obtained by PyTorch Lifestream models
- `recommendation_pipeline` notebook contains the pipeline of generating recommendations according to model predictions, specifically for Gowalla dataset
- `flask_api` folder contains the Flask API wrapper for recommendation pipeline
