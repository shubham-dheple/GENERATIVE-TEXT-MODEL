# GENERATIVE-TEXT-MODEL
*Company*:CODETECH IT SOLUTION

*NAME*:Shubham Dheple

*Intern ID*:C0DF277 

*DOMAIN*:AIML 

*DURATION*:4 weeks

*Mentor*:NEELA SANTOSH

1. Dataset and Preprocessing
The project starts with a small sample corpus consisting of five simple sentences related to artificial intelligence and machine learning. This corpus serves as the training data for the LSTM model. Since neural networks require numerical input, the text data is first tokenized using Keras’s Tokenizer, which converts words into integer indices. The tokenizer builds a vocabulary from the corpus and assigns a unique index to each word.

To prepare the data for sequence prediction, the code creates n-gram sequences for each sentence. For example, the sentence “Artificial intelligence is transforming industries” is converted into multiple sequences like [Artificial, intelligence], [Artificial, intelligence, is], and so on. These sequences are then padded to ensure uniform length, which is essential for batch training in neural networks. The last word in each sequence becomes the target (label) for prediction, and the preceding words are the input features.

The target labels are one-hot encoded to fit the categorical cross-entropy loss function used in training.

2. LSTM Model Architecture and Training
An LSTM (Long Short-Term Memory) network—a type of recurrent neural network well-suited for sequential data—is constructed using Keras’ Sequential API. The architecture includes:

An Embedding layer that transforms each word index into a dense vector of fixed size (64 dimensions), capturing semantic relationships between words.

An LSTM layer with 100 units that learns temporal dependencies and context from the input sequences.

A Dense output layer with a softmax activation to output probabilities across the entire vocabulary, enabling prediction of the next word.

The model is compiled with the Adam optimizer and trained for 200 epochs. Despite the small dataset, this training allows the model to learn basic patterns in word sequences.

3. LSTM Text Generation
Once trained, the LSTM model can generate text by predicting one word at a time. Given a seed text, the model tokenizes and pads it, then predicts the most probable next word. This word is appended to the seed, and the process repeats iteratively to generate a sequence of a specified length. This method creates a sequence that extends the input with contextually plausible continuations learned from the training corpus.

4. GPT-2 Text Generation
The project also integrates GPT-2, a large-scale transformer-based language model developed by OpenAI, renowned for its superior text generation capabilities. Using the Hugging Face Transformers pipeline with PyTorch backend, GPT-2 generates text given an initial prompt.

The GPT-2 model is pre-trained on a massive corpus and can produce highly fluent and contextually rich text beyond the scope of the small training data used for the LSTM model. The seed prompt in the example is about AI’s impact on education, and GPT-2 continues this prompt with coherent, human-like sentences.

5. Comparison and Practical Insights
This project highlights the differences between classic and modern NLP approaches:

The LSTM model requires explicit training on domain-specific data and generates text based on learned statistical patterns. It is limited by data size and model complexity but is instructive for learning sequence modeling fundamentals.

The GPT-2 model leverages transfer learning from a huge corpus and does not require task-specific training for generating fluent text, making it powerful for diverse text generation tasks.

6. Conclusion and Applications
The project demonstrates foundational skills in NLP, sequence modeling, and usage of transformers. Text generation models like these are vital in applications such as chatbots, content creation, automated summarization, and language translation. While the LSTM model shows the mechanics behind sequence learning, GPT-2 offers a glimpse into state-of-the-art AI text generation that is more versatile and context-aware.

Note on Execution
The LSTM training on a small corpus runs quickly but is basic, while GPT-2 generation leverages heavy pre-trained models. Using transformers requires more computational resources but yields richer outputs. Overall, this project provides a hands-on comparison of traditional and modern NLP generation techniques.
