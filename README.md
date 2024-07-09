# LLM RAG Chatbot (with only CPU)

## :monocle_face: Description

In this project, we deploy a **LLM RAG Chatbot** with **Langchain** on a **Streamlit** web application using only **CPU**. The LLM model is designed to extract relevant information from external documents. We have used the quantized version of **Llama-2-7B** with the **GGML** quantization approach, enabling it to run on CPU processors.

Traditionally, LLMs have relied on prompts and the training data on which the model was trained. However, this approach has limitations in terms of knowledge, especially when dealing with large datasets that exceed token length constraints. To address this challenge, RAG (Retrieval Augmented Generation) enriches the LLM with new and external data sources.

Before demonstrating the Streamlit web application, let's walk through the details of the RAG approach to understand how it works. The **retriever** acts like an internal search engine: given a user query, it returns a few relevant elements from the external data sources. Here are the main steps of the RAG system:

1. **Document Chunking and Embedding:** Split each document of our knowledge base into chunks and obtain their embeddings. When embedding documents, use a model that accepts a certain maximum sequence length (max_seq_length).

2. **Vector Database Storage:** Once all chunks are embedded, store them in a vector database. When a user types a query, it gets embedded by the same model used earlier, and a similarity search returns the top_k closest chunks from the vector database. This requires:
   - A metric to measure the distance between embeddings (e.g., Euclidean distance, Cosine similarity, Dot product).
   - A search algorithm to find the closest elements (e.g., Facebook's FAISS). Our model works well with cosine similarity.

3. **Context Aggregation:** The content of the retrieved documents is aggregated into the "context", which is then combined with the query into the "prompt" and fed to the LLM to generate answers.



To achieve good accuracy with LLMs, it is important to understand and choose each hyperparameter carefully. The LLM's decoding process involves two main types of decoding: **greedy** and **sampling**. Greedy decoding selects the token with the highest probability at each step, while sampling decoding introduces variability by selecting a subset of potential output tokens randomly.

When using sampling decoding, two additional hyperparameters impact the model's performance:

- **top_k:** An integer ranging from **1** to **100** representing the k tokens with the highest probabilities. For example, if we want to predict the next token and have three possibilities with probabilities [0.23, 0.12, 0.30], setting top_k = 2 will select the two tokens with the highest probabilities.

- **top_p:** A decimal ranging from **0.0** to **1.0**. It selects a subset of tokens with cumulative probabilities equal to top_p. Using the same example, setting top_p = 0.55 will select tokens with cumulative probabilities below 0.55.

- **temperature:** This parameter ranges from **0** to **2** and adjusts the probability distribution of output tokens. Lower values make higher probability tokens more likely, resulting in predictable responses, while higher values make probabilities converge, encouraging creativity.

Another parameter to consider is the memory needed to run the LLM. For a model with N parameters and full precision (fp32), the memory required is N x 4Bytes. Using quantization, this memory requirement is divided by the new precision (e.g., fp16 reduces memory by half).

## :rocket: Repository Structure

The repository contains the following files and directories:
- **app:** Contains the Streamlit code for the **LLM RAG Chatbot** web application.
- **Dockerfile:** Contains instructions to build the Docker image.
- **images:** Contains all images used in the README file.
- **requirements.txt:** Lists all the packages used in this project.

## :chart_with_upwards_trend: Demonstration

In this section, we demonstrate the Streamlit web application. Users can ask any question, and the chatbot will provide answers.

To deploy the Streamlit app with Docker, use the following commands:

```sh
docker build -t streamlit .
docker run -p 8501:8501 streamlit
```

To view the app, browse to [http://0.0.0.0:8501](http://0.0.0.0:8501) or [http://localhost:8501](http://localhost:8501).# LLM-Chatbot
