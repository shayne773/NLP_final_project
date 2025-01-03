# GUSUM: Graph-Based Unsupervised Summarization

## Overview
This repository contains an implementation of the **GUSUM** (Graph-Based Unsupervised Summarization) algorithm, an unsupervised extractive summarization model that utilizes graph-based techniques to extract key sentences from a document. The project enhances the original GUSUM model by introducing modifications such as improved sentence feature weighting, alternative centrality measures, and diversification in sentence selection to improve the quality of summaries, as evaluated by ROUGE metrics.

## Features
The enhanced GUSUM model includes the following key features:
- **Sentence Feature Scoring**: Four core features—sentence length, position, proper nouns, and numerical tokens—are used to score each sentence and define its importance in the summary.
- **Sentence Embeddings**: Sentence embeddings are generated using SentenceBERT to calculate semantic similarity between sentences.
- **Graph Creation**: A graph is constructed where each node represents a sentence, and edges represent semantic relationships based on cosine similarity.
- **Centrality Calculation**: A new centrality metric incorporates forward and backward edges to prioritize sentences supported by earlier content in the document.
- **Diversity in Sentence Selection**: Sentences with high cosine similarity are excluded to ensure that selected sentences are thematically diverse.
- **TF-IDF Keyword Ranking**: TF-IDF is integrated as an additional feature for keyword ranking, further improving ROUGE scores.

## Dataset
The dataset used for this research is the **CNN/DailyMail** corpus, consisting of over 300,000 news articles. For this study, we used 40,000 CNN articles (30,000 for training and 10,000 for testing). The dataset includes human-expert summaries, which are used to evaluate the quality of the generated summaries using ROUGE metrics.

## Enhancements
Several modifications have been made to the original GUSUM framework to enhance performance:
1. **Weight Adjustments**: Feature weights were adjusted to prioritize important features, such as sentence position and numerical tokens.
2. **Diversity in Sentence Selection**: Sentences are selected based on both their importance and diversity, avoiding highly similar sentences in the summary.
3. **Modified Centrality**: Forward and backward edges are given different weights to reflect the positional importance of sentences within the document.
4. **TF-IDF**: Term Frequency-Inverse Document Frequency (TF-IDF) is used to adjust feature scores, improving ROUGE evaluation scores.
