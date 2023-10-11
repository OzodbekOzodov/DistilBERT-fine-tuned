[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/R1vgPUT1)
# Fine-Tuning BERT for Sentiment Analysis of Monetary Policy Speeches: A Study Using Central Bank Transcripts

## How to use and navigate through this repository

Clone the repo using  

`$ git clone https://github.com/iame-uni-bonn/final-project-OzodbekOzodov`

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


## ANSWERS

1. List five different tasks that belong to the field of natural language processing.
Sentiment analysis, machine translation, text generation, named entity recognition,text summarization.   


2. What is the fundamental difference between econometrics/statistics and suprevised machine learning
Econometrics/statistics focuses on using sample, making inference for population based on sample, testing the hypotheses and tries to establish causal link under certain assumptions. Supervised machine learning, on the other hand, focuses on learning the patterns from input data, and tries to predict with best accuracy possible.

3. Can you use stochastic gradient descent to tune the hyperparameters of a random forrest. If not, why?   

SGD is not suitable for Random forest. Random forest is an ensemble model - bootstraped decision tree, and it has hyperparameters like depth, number of trees, and SGD rules them out. SGD is designed for models where small changes in parameters produce smooth changes in performance. In contrast, random forests have discrete and non-smooth hyperparameters, making SGD ineffective for tuning them.    


4. What is imbalanced data and why can it be a problem in machine learning?    

When the number of labeled data for each class is significantly different from the others, there is imbalance in classes. This leads to underlearning of certain class, since the model does not have enough data to learn from that class. As a result, model can not capture and generalize the properties of the class, and underperform when predicting the classes.

5. Why are samples split into training and test data in machine learning?   

Test data helps see if the model can do any good when unseen data is thrown into the model. When the model is trained with all data, testing it with it's own chunk or whole fraction does not make much sense, since the model already learned from exactly that data. To see what potentially happens in production, leaving the test data aside and testing the model on it is useful to see what the model would do with new unseen data. Train-test split is one of the most important steps for evaluating the model.

6. Describe the pros and cons of word and character level tokenization.    

Word-level tokenization quickly processes common units of meaning but struggles with out-of-vocabulary words and needs more memory for large vocabularies. Meanwhile, character-level tokenization can manage any word, even misspelled or very subject specific ones, with a smaller vocabulary size. However, it often results in longer sequences, slowing processing, and misses the higher-level semantic meaning of words.

7. Why does fine-tuning usually give you a better performing model than feature extraction?    

Fine-tuning usually leverages already the big enough models, and adjusts certain weights and biases of that big model. While feature extraction leverages the knowledge from the pre-trained model only as a fixed feature extractor, which makes it less flexible with even slightly different data.

8. What are advantages over feature extraction over fine-tuning?    

Faster training, memory and computational efficiency, amount of data required. Generally, fine-tuning goes over many more parameters, measures and changes them, and this takes a lot of resources. Feature extraction can also mitage the risk of overfitting.

9. Why are neural networks trained on GPUs or other specialized hardware?    

Neural nets require a lot of computational power, and there are only a few types of hardwares that can do it, like CPUs, GPUs and TPUs. GPUs are generally better and cheaper than CPUs with special ways of parallel processing. Also frameworks like Pytorch and TensorFlow are designed to utilize GPUs more efficiently.

10. How can you write pytorch code that uses a GPU if it is available but also runs on a laptop that does not have a GPU?    

We can set a condition that checks if any GPU is aavailable, and uses it if the condition is met, or use CPU otherwise. Example: # set the device based on availability of hardware device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   
    

model = torch.nn.Linear(X, Y).to(device) # Move the model to the chosen device     
    
data = torch.randn(X,Y).to(device) # Move the data to the chosen device    


11. How many trainable parameters would the neural network in this video have if we remove the second hidden layer but leave it otherwise unchanged.   


When the second layer was removed from the neural network, it led to the elimination of all the weights and biases associated with that layer, specifically the weights connecting the first layer to the second and the second to the third, as well as the biases of the second layer itself. However, a new direct connection between the first and third layers was established, introducing a fresh set of weights. While this new connection added parameters, the overall reduction stemmed from the loss of two sets of weights (from both sides of the removed layer) and the associated biases. In essence, the act of layer removal simplified the model by reducing its trainable parameters, making it a less complex representation.    

12. Why are nonlinearities used in neural networks? Name at least three different nonlinearities.    

Non-linearity is actually one of the things that allows neural networks to learn basically anything. Certain nonlinearities help in propagating gradients during backpropagation, facilitating the way of learning that other traditional models can not learn. Some of the nonlinearities are Rectified linear unit, TanH and Sigmoid.

13. Some would say that softmax is a bad name. What would be a better name and why?    

Since the nature of softmax is to take the values, ensure they are positive, and normalize them, I'd also call it positive normalization function.    


14. What is the purpose of DataLoaders in pytorch?    

The DataLoaders in Pytorch has several purposes: loading the data from various sources in a format and way that PyTorch can easily handle without taking many steps prior to training actual model, also it facilitates shuffling, batching and parallel transformation of data.    


15. Name a few different optimizers that are used to train deep neural networks     
   
Adam, AdamW, SGD    


16. What happens when the batch size during the optimization is set too small?    


When the batch size during optimization is set too small, the training can become noisy leading to less stable convergence, the process might take longer since fewer examples are processed at a time, the model may not generalize well due to the frequent and noisy updates, and the computational capabilities, especially of GPUs, might not be fully utilized.    


17. What happens when the batch size diring the optimization is set too large?    

The model requires more memory and converges faster, but might local get stuck in local minima.

18. Why can the feed-forward neural network we implemented for image classification not be used for language modelling?    

The data used in language tasks have sequential nature, making the feed-forward networks struggle dealing with sequential data. Feed-forward models does not have as big of a squential memory.   


19. Why is an encoder-decoder architecture used for machine translation (instead of the simpler encoder only architecture we used for language modelling)    

An encoder captures source language semantics, and a decoder generates the target language. A simple encoder can't handle the variability of translations.    

20. Is it a good idea to base your final project on a paper or blogpost from 2015? Why or why not?   

Most of the today's powerful NLP tools, techniques and frameworks are built based on transformer architecture, which was originally announced in 2017. Computer linguistics has experienced a breakthrough after the attention-based models, and anything from old era have a chance that they have been improved after 2017.   


21. Do you agree with the following sentence: To get the best model performance, you should train a model from scratch in Pytorch so you can influence every step of the process.    

Not necessarily. Although training from scratch sounds about right, fine-tuning and transfer learning might be less time and resource consuming, and lead to as good performance, depending on the specific task, data and available resources.   

22. What is an example of an encoder-only model?   

BERT, DistilBERT or any other BERT-based models.

23. What is the vanishing gradient problem and how does it affect training?    

It's when gradients in a deep network become too small, causing early layers to learn very slowly or not at all. This makes training harder and slower.

24. Which model has a longer memory: RNN or Transformer?   

Transformers, especially with attention mechanisms, can capture longer-term dependencies more effectively than typical RNNs.

25. What is the fundamental component of the transformer architecture? The attention mechanism, which allows the model to focus on different parts of the input differently.  
