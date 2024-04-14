import dspy
import time
import streamlit as st
import openai
from typing import List, Union, Optional
import qdrant_client
from dspy import Predict
from dotenv import load_dotenv
import os
from llama_index.core import Settings, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

load_dotenv() 

openai_client = openai.Client()

# Configuration parameters for the vector store and OpenAI embeddings
collection_name = "llama_parse_prudential_health_both_pdfs"
openai_embedding_model = "text-embedding-3-small"

q_client = qdrant_client.QdrantClient(

    host="localhost",
    port=6333
)

# Custom retrieval model class, integrates Qdrant with DSPy
class DSPythonicRMClientQdrantCustom(dspy.Retrieve):
    def __init__(self, q_client, collection_name: str,embedding_model ,k: int = 3):
        super().__init__(k=k)
        self.q_client = q_client
        self.collection_name = collection_name
        self.embedding_model = embedding_model

    def forward(self, query: str, k: Optional[int] = None) -> dspy.Prediction:
        query_embedding = openai_client.embeddings.create(
            input=[query],
            model=self.embedding_model
        ).data[0].embedding

        # Search for relevant documents in the Qdrant vector store
        results = self.q_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k if k else self.k
        )

        import json # hacky way to get this part to work

        # Process and return the search results
        return dspy.Prediction(passages=[json.loads(point.payload["_node_content"])["text"] for point in results])

# Configure the dspy settings with custom retrieval model and GPT-4 model
llm = dspy.OpenAI(model='gpt-4-turbo')
retriever = qdrant_retriever_model = DSPythonicRMClientQdrantCustom(
                                                        q_client = q_client,
                                                        collection_name = collection_name, 
                                                        embedding_model=openai_embedding_model, 
                                                        k=5)

dspy.settings.configure(lm=llm, rm=retriever)

class GenerateAnswer(dspy.Signature):
    """Answer questions with descriptive answers, you are outputting to customers
        of an insurance company, so helpfully answering the question is essential. 
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="""make sure the customer's query is 1. grounded in the response and appropriately 
                              answered and would be happy to have read your accurate and comprehensive response. Don't be over 2 sentences though
                              If the answer is not grounded in the context, and you are not sure <30% confident in your answer, 
                              please go with your intuition on your pre-trained knowledge to answer and let the user know the most likely product it is found in eg- PRUHealth Guardian 
                            Critical Illness Plan Series
                              """)
    print("Class 1 created")

# ***************************** Implementing Multi-Hop *****************************
class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question.
        If part of the query is answered, try focus on the part that isn't with a completely differently framed query
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


from dsp.utils import deduplicate

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = DSPythonicRMClientQdrantCustom(q_client, collection_name = collection_name,
                                                embedding_model=openai_embedding_model, 
                                                 k=passages_per_hop)
        self.generate_answer = Predict(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)
    
# rag = RAG()
rag = SimplifiedBaleen()
rag.load(path="dspy_program/compiled.txt")


class BasicQA(dspy.Signature):
    """Answer the question with a short answer."""

    question = dspy.InputField()
    answer = dspy.OutputField()

qa = dspy.Predict(BasicQA)


# Define a Streamlit app for a healthcare chatbot

st.title('Prudential Plc Healthcare Chatbot April 2024')
st.sidebar.title("Chat History")

task = st.sidebar.radio(
    "Simple QA or RAG?",
    ["Simple QA", "RAG"]
)

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

txt = st.chat_input("Say something...")
if txt:
    st.session_state['chat_history'].append("User: "+txt)
    chat_user = st.chat_message("user")
    chat_user.write(txt)
    chat_assistant = st.chat_message("assistant")
    with st.status("Generating the answer...") as status:
        tms_start = time.time()
        if task == "Simple QA":
            ans = qa(question=txt).answer
        elif task == "RAG":
            ans = rag(question=txt).answer
        chat_assistant.write(ans)
        st.session_state['chat_history'].append("Assistant: "+ans)
        tms_elapsed = time.time() - tms_start
        status.update(
            label="Answer generated in %0.2f seconds." \
                % (tms_elapsed), state="complete", expanded=False
        )
    st.sidebar.markdown(
        "<br />".join(st.session_state['chat_history'])+"<br /><br />", 
        unsafe_allow_html=True
        )
