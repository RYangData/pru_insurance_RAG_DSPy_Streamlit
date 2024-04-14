# Insurance Brochure 2-Document RAG with DSPy and Streamlit

## Installation

To get started with this project, you'll need to install the required Python packages. You can install these packages using pip. Run the following command in your terminal:

```bash
pip install -r requirements.txt
```

## Environment Variables

This project requires setting up environment variables to securely store API keys. Follow these steps to set up your environment variables:

1. Create a `.env` file in the root directory of the project.
2. Open the `.env` file and add the following lines:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
```
Please refer to here for LLAMA cloud API key - https://github.com/run-llama/llama_parse.

Replace `your_openai_api_key_here` and `your_llama_cloud_api_key_here` with your actual API keys.

## Running the Application

To run the application, you first need to start the Qdrant server using Docker. Run the following Docker command in your terminal to set up and start the Qdrant service:

```bash
docker run -p 6333:6333 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

This command will start a Qdrant Docker container, mapping port 6333 for access and using a volume mount for persistent storage.

After ensuring that the Qdrant server is running, proceed with the following steps to parse documents and run the Streamlit app:

### Indexing PDFs

Run the `index_pdfs.py` script to parse two documents and perform vector indexing:

```bash
python3 index_pdfs.py
```
Parsing is done with llama_parse: https://github.com/run-llama/llama_parse

### Running the Streamlit App

After indexing the documents, run the Streamlit app:

```bash
streamlit run app.py
```

In the app interface, there are two options on the left-hand side (LHS):

- **Simple QA**: Engages in simple chat with GPT-4.
- **RAG (Retrieval-Augmented Generation)**: Enables document-based Q&A on the two indexed documents.

## Future Work

- Conduct more in-depth experimentation and evaluation of the RAG component.
- Compile a full-scale evaluation with DSPy and experiment with additional modules, including fine-tuning.
- Containerise and deploy the application on Google Cloud Platform's Cloud Run.

## Additional Notes
- The `experims_llama_parse_and_dspy.ipynb` file found in the notebooks folder contains all the experimentation and tailoring/prompting of the parsing + DSPy framework. Please refer to here for more of an idea of how I created the app. 
