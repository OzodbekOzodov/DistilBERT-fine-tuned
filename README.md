[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/R1vgPUT1)
# Fine-Tuning BERT for Sentiment Analysis of Monetary Policy Speeches: A Study Using Central Bank Transcripts

### Please, pay attention that you can not see the commit history and the path the model is developed, because the original code I wrote was in private, organizational repository, and I took relevant parts to this repository

## How to use and navigate through this repository

Clone the repo using  

`$ git clone https://github.com/OzodbekOzodov/DistilBERT-fine-tuned

## Setting up the environment for dependency handling  


Navigate to the local folder and run this script on terminal to install the environment  

`$ conda env create -f environment.yml`

Staying in the same directory, run this to activate the environment  

`$ conda activate policy-speech-classification`

## Folder and code structure

├── src/    
│ ├── Data/   
│ │ └── Filtered_labels.csv   
│ │   
│ └── Output/   
│ └── loss-curves.jpg   
│   
├── 1_Fine-tuning_BERT.ipynb # Main notebook   
├── 2_using_model_from_web.ipynb # Testing the model   
├── environment.yml # Conda environment file   
└── README.md # Documentation   

## Coding Conventions   

This repository follows a blend of coding styles and best practices:  


- PyTorch and Hugging Face: Our deep learning components are grounded in PyTorch standards and the conventions of Hugging Face's Transformers library   

- OSE Deep Learning Class Style: Coding practices from the OSE Deep Learning class are also infused throughout, ensuring consistency and alignment with academic standards   

- Readability: The name of the variables, parameters are set to be as clear as possible making it intuitive to understand what role the variable plays in the code   

- Comments: Vital sections of the code are accompanied by comments, offering context or explaining complex operations.   





## Abstract:
The interpretation and sentiment analysis of central bank policy speeches represent a vital task in understanding monetary policy signals. This project aims to apply both cutting-edge language models and traditional neural networks to extract sentiment scores from monetary policy speeches across over 100 countries spanning more than a decade. Specifically, the project focuses on fine-tuning the BERT (Bidirectional Encoder Representations from Transformers) model using labeled parts of central bank speeches, aiming to specialize the model in understanding the unique intricacies of monetary policy language.

## Objectives:

Fine-Tuning Language Models: The project's primary objective is to adapt existing Language Model (LM) architectures such as DistilBERT for the specific task of sentiment analysis in central bank policy speeches. This involves using carefully annotated and randomly selected parts of speeches to fine-tune the model.

Sentiment Analysis: Key to the project is extracting sentiment scores that accurately reflect the policy stance in central bank communications. The analysis will strive to encapsulate the nuanced and complex rhetoric used in these speeches.

Evaluation Metrics: Rigorous evaluation methods, including precision, recall, F1-score, accuracy will be employed to assess the model's success in classifying sentiments within monetary policy contexts.

Adaptation to Monetary Policy Language: Given the specialized language and terminology used in monetary policy talks, an essential objective is to develop a model that understands and processes this unique domain language.

Methodology:

Data Acquisition and Preprocessing: Transcripts of central bank speeches from over 100 countries will be collected, and selected parts will be annotated for sentiment analysis. Preprocessing steps such as tokenization and stop-word removal will be implemented.

Fine-Tuning BERT: The BERT model will be specialized through fine-tuning using the labeled central bank speech data. This fine-tuning aims to make the model adept at deciphering the monetary policy lexicon and rhetoric.

Sentiment Analysis Execution: Leveraging the fine-tuned BERT model, sentiment scores will be derived from the speeches. This involves careful analysis of language patterns, tone, and context within the monetary policy domain.
