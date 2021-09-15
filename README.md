# Bridgei2i's Automated Headline and Sentiment Generator

Automated identification, summarization, and entity-based sentiment analysis of mobile technology articles and tweets.

**Problem:**
*  Develop an intelligent system that could first identify the theme of tweets and articles.
* If the theme is mobile technology then it should identify the sentiments against a brand (at a tweet/paragraph level).
* We would need a one-sentence headline of a max of 20 words for articles that follow the mobile technology theme. A headline for tweets is not required.

## 0. Data Description

For this task, we received a mix of 4000 non-mobile tech and mobile tech tweets and articles each, with their labels of mobile_tech_tag as well as the headlines for the articles. No data was given for entity-based sentiment analysis. 

## 1. Pipeline

![Pipeline](https://github.com/mukul54/inter-iit-bridgei2i/blob/main/_assets/pipeline.png)

We created an end-to-end pipeline that goes from the input to the output while ensuring good efficiency as well as code scalability. 

### 1.1  Preprocessing

We created a pre-processing pipeline to remove all the useless portions of the text and leave the useful features. For that, we did the following:

*  Remove the Hyperlinks and URLs
*  Segment the hashtags
*  Demojify the emojis
*  Remove punctuations
*  Convert the text into lower-case

After we passed our dataset through the preprocessing pipeline, we noticed that most of the tweets and articles were repeated multiple times.

For the removal of duplicate sentences, we developed a unique graph-based clustering algorithm using Levenshtein distance, since there were a lot of examples that were different because of the presence of some of the gibberish or very uncommon words. We were unable to identify them just by using existing duplicate methods in pandas. So we developed a clustering algorithm that clusters similar words with Lavestein distance less than a certain threshold and then one datapoint form each cluster is
selected as a unique example.

On removing the duplicates, we noticed that only 10% of the dataset had unique text and hence were left with approximate 400 tweets and articles.

### 1.2 Language Detection

For the language detection module, we built a Bidirectional LSTM based model which was trained on a self-made dataset that contained Hinglish and English words. The model learnt to identify Hinglish words to an accuracy of 93%. 

Language detection helps us identify the language of the tweet or the article so that we can divide them into code-mixed, pure-Indic language and English sentences.  


### 1.3  Transliteration and Translation

One of the biggest challenges in our dataset was to deal with code-mixed languages. Code-mixed languages are pretty common in countries with bilingual and multilingual culture. To deal with code-mixed language, we thought of transliterating the sentences to their intended language so that the code-mixed language is converted into a pure-Indic language. 

To create a model for transliteration, we first needed to collect data for it. We collected our data from the following sources:

* Xlit-Crowd: Hindi-English Transliteration Corpus
These pairs were obtained via crowdsourcing by asking workers to
transliterate Hindi words into the Roman script. The tasks were done on
Amazon Mechanical Turk and yielded a total of 14919 pairs.
* NeuralCharTransliteration: Data is created from the scrapped Hindi songs lyrics.
* Xlit-IITB-Par: This is a corpus containing transliteration pairs for Hindi-English. These pairs were automatically mined from the IIT Bombay English-Hindi Parallel Corpususing the Moses Transliteration Module. The corpus contains 68,922 pairs.
* BrahmiNet Corpus: 110 language pairs
* Hindi word transliteration pairs
* Dakshina dataset: The Dakshina dataset is a collection of text in both Latin and native scripts for 12 South Asian languages. For each language, the dataset includes a large collection of native script Wikipedia text, a romanization lexicon which consists of words in the native script with attested romanizations, and some full sentence parallel data in both a native script of the language and the basic Latin alphabet.

We collated the above datasets, cleaned them and created a final data file to train our model(available in the data folder). Once the data-file was made, we trained a transformer model on it, since the parallelism makes transformers faster than RNNs and this task needed speed 

### 1.4  Classification of mobile_tech text

For the classification of the input data into tech and non-tech, we used a stacked BiLSTM on the preprocessed text. The reason for using a BiLSTM as opposed to a transformer-based model was that the BiLSTM model gave similar scores as the transformer-based model while also being faster.

In order to help the model learn the given data distribution and generalize to newer out-of-distribution examples as well, 80% of the given data was concatenated with the scraped dataset, and we used this combined data for training.

### 1.5  Brand Identification & Aspect based sentiment analysis
For brand identification of mobile tech tweets and articles, we are using a dictionary-based approach and the accuracy can
further, be increased by using a transformer for NER (Named Entity Recognition).

For aspect based sentiment analysis , a transformer based model is used for feature extraction which is further fed into a GRU model to get sentiment of each word of the sentence. Finally we will cherry-pick the sentiments of the brands.

Due to their inherent capability in semantic alignment of aspects and their context words, attention mechanism and Convolutional Neural Networks (CNNs) are widely applied for aspect-based sentiment classification. However, these models lack a mechanism to account for relevant syntactical constraints and long-range word dependencies, and hence may mistakenly recognize syntactically irrelevant contextual words as clues for judging aspect sentiment. To tackle this problem, we are using a [Graph Convolutional Network (GCN)](https://arxiv.org/abs/1909.03477) over the dependency tree of a sentence to exploit syntactical information and word
dependencies. Based on it, an aspect specific sentiment classification framework is raised.




### 1.6  Heading generation
In this task, a summary of a given article/document is generated when passed through a network. There are 2 types of summary generation mechanisms:

1. ***Extractive Summary:*** the network calculates the most important sentences from the article and gets them together to provide the most meaningful information from the article.
2. ***Abstractive Summary***: The network creates new sentences to encapsulate the maximum gist of the article and generates that as output. The sentences in the summary may or may not be contained in the article. 

Here we will be generating ***Abstractive Summary***. 

- **Data**:
    - We are using the News Summary dataset available at [Kaggle](https://www.kaggle.com/sunnysai12345/news-summary)
    - This dataset is the collection created from Newspapers published in India. We also have scrapped mobile tech articles from
https://gadgets.ndtv.com/mobiles with headlines.



- **Language Model Used**: 
    - This notebook uses one of the most recent and novel transformers model ***T5***. [Research Paper](https://arxiv.org/abs/1910.10683)    
    - ***T5*** in many ways is one of its kind transformers architecture that not only gives state of the art results in many NLP tasks, but also has a very radical approach to NLP tasks.
    - **Text-2-Text** - According to the graphic taken from the T5 paper. All NLP tasks are converted to a **text-to-text** problem. Tasks such as translation, classification, summarization and question answering, all of them are treated as a text-to-text conversion problem, rather than seen as separate unique problem statements.

## 2. File Structure: How to run?

### USAGE: Transliteration and Translation
In order to translate a text you need to follow the following mentioned steps:
- `cd submission`
- change the `PATH` variable in the `__name__ == "__main__"` block of the `language_convert.py`
- coloumn name of the text to be converted should be `Text`.
- `python language_convert.py` 
### USAGE: Text Classification
This model have directly been trained and evaluated in [final notebook](https://github.com/mukul54/inter-iit-bridgei2i/blob/main/notebooks/i2i_submission.ipynb).
### Brand Identification
This model have directly been trained and evaluated in [final notebook](https://github.com/mukul54/inter-iit-bridgei2i/blob/main/notebooks/i2i_submission.ipynb).
### USAGE: Aspect Based Sentiment Analysis
Spacy and Language models installation:

    pip install spacy
    python3 -m spacy download en
Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.<br />
If graph and tree files don't exist in the dataset,then graph and tree data are generated with:

    python3 dependency_tree.py
    python3 dependency_graph.py
Trained with:

    python train.py --model_name asgcn --dataset interiit --save True
Infer with:

    python3 infer.py

## 3. Contributions
1. We developed an **end-to-end, scalable and unique pipeline for the identification,
summarization and entity based sentiment analysis** of the tweets and articles.
2. We trained the character based transliteration model from scratch for the conversion of code-mix language (Hindi+English for our case) to the pure language(Hindi for our case), which is **highly scalable, reliable and fast** (our model is twice as fast as Google’s API) that can be scaled to any other code-mix language as well.
3. We developed a unique dictionary based approach for the brand identification.
4. We finetuned a transformer based model forheadline generation and used a  GCN based model for Aspect Based Sentiment Analysis.
5. We developed different datasets by scrapping from various news sources and twitter and then cleaned and preprocessed them for training the models on different tasks.
6. Our preprocessing pipelines consist of a unique graph based clustering algorithm
for the identification of duplicate examples based on Levenshtein distance.

## 4. Future Prospects
- The translation part of the pipeline can be scaled to multiple code-mixed language.
- Since it was mentioned that in case of various models of a brand, we need to identify then only if they apperar with their brand name, for example galaxy need to be identified as brand only if it is mentioned in the text with Samsung, so We can develop a model to learn the reverse mapping between brand and model automatically with availability of more data.
- We availability of more data the pipeline can be scaled to other domain like fashion, transport as welll.
- The entire pipeline can be deployed for commercial use.

## 5. Acknowledgements and References
For different part of the pipeline we have used various open source libraries, datasets and availabe codebase. We are thanksful to the authors of those libraries codebases and datasets.
- [ASGCN Codebase](https://github.com/GeneZC/ASGCN)
- [ASGCN Paper](https://arxiv.org/abs/1909.03477)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- [T5 Hugging face transformer](https://huggingface.co/transformers/model_doc/t5.html)
- [News Summary Dataset Kaggle](https://www.kaggle.com/sunnysai12345/news-summary)
- [MarianMT](https://huggingface.co/transformers/model_doc/marian.html)
- [Pytorch](https://pytorch.org/)
