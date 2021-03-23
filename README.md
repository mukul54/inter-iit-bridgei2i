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
### 1.6  Heading generation


## 2. File Structure: How to run?
## 3. Contributions
## 4. Future Prospects
## 5. Acknowledgements
