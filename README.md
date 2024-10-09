# vqa-for-remote-sensing
# Remote Sensing Visual Question Answering (RSVQA) with ViLT

This repository contains the implementation of a Visual Question Answering (VQA) system tailored for remote sensing imagery using the **ViLT** (Vision-and-Language Transformer) architecture. The model is fine-tuned to answer natural language questions related to satellite images, combining both visual and textual inputs.

## Project Overview

The goal of this project is to train a ViLT-based model to understand questions about remote sensing images and provide accurate answers by leveraging both visual and textual information. The ViLT model is fine-tuned with a custom dataset that includes questions, images, and encoded answers, enabling it to predict answers for unseen visual questions.

## Features

- **Remote Sensing Visual Question Answering (RSVQA)**: Tailored for satellite images and geo-spatial questions.
- **ViLT Model**: Fine-tuned version of the Vision-and-Language Transformer for answering questions based on image and text inputs.
- **Preprocessing Pipeline**: Images are converted to embeddings, and text questions are tokenized and preprocessed for the ViLT model.
- **Answer Encoding**: Categorical answers are encoded using `LabelEncoder` from `sklearn` to facilitate model learning.
- **Training and Testing**: The dataset is split into training and testing sets, with 20% of the data used for model evaluation.
- **Transfer Learning**: Pre-trained weights are used for initializing the ViLT model to leverage prior knowledge and speed up training.

## Data Preparation

The dataset consists of three main components:
1. **Questions**: Natural language questions related to remote sensing imagery.
2. **Images**: Satellite images in `.tif` format, preprocessed into embeddings for efficient input to the model.
3. **Answers**: Text-based answers, encoded as numerical labels using `LabelEncoder`.

### Steps:

1. **Load Processed Data**: 
   Load the processed question-answer pairs and image embeddings from a CSV file into a Pandas DataFrame. The answers are encoded into numerical values using `LabelEncoder` to enable easier model training.

2. **Data Splitting**:
   Use `train_test_split` to divide the dataset into training (80%) and testing (20%) subsets.

3. **VQADataset Class**:
   A custom `VQADataset` class is defined to efficiently load and preprocess data in batches using PyTorch's `Dataset` class.

## Model Architecture

The ViLT model architecture is composed of both a transformer for text and visual input processing, along with an output layer tailored to the specific number of answer classes in the dataset. The model is initialized with pre-trained weights to speed up training and improve performance.

### Training Process

- The model is fine-tuned using the AdamW optimizer with a learning rate of `5e-5`.
- The training loop runs for 3 epochs, during which image embeddings and tokenized questions are passed through the ViLT model.
- Cross-entropy loss is used as the objective function, calculated based on the predicted answer probabilities and the encoded correct answer.
- Gradients are calculated and used to update model weights using backpropagation.
- Loss values are printed periodically to monitor the training progress.

## How to Run the Project

Ensure you have the following installed:

- Python 3.7+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Pandas

