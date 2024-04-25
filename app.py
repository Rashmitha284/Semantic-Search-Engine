from flask import Flask, render_template, request, session, redirect, url_for
from flask_session import Session
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import csv

# Initialize the Flask app
app = Flask(__name__)

# Configure Flask session to use the filesystem
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Increase CSV field size limit
csv.field_size_limit(100000000)

# Load the embedded DataFrame
df_emd = pd.read_csv(r"C:\Users\justin\Downloads\search engine project\embeddings_data.csv")

# Instantiate ChromaDB PersistentClient and define embedding function
chroma_client_1 = chromadb.PersistentClient(path=r"C:\Users\justin\Downloads\search engine project\vector_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")

# Get or create the collection
collection_1 = chroma_client_1.get_or_create_collection(name="my_collection_1", embedding_function=sentence_transformer_ef, metadata={"hnsw:space": "cosine"})

@app.route('/')
def index():
    # Pop message from session (if any) to display in the template
    message = session.pop('message', None)
    return render_template('index.html', message=message)

@app.route('/search', methods=['POST'])
def search():
    # Get query from the form
    query = request.form.get('query', '').strip()

    if not query:
        # If query is empty, set message and redirect to index
        session['message'] = 'Please enter a query.'
        return redirect(url_for('index'))

    # Execute query against the collection
    results = collection_1.query(
        query_texts=[query],
        n_results=10,
        include=['documents', 'distances', 'metadatas']
    )

    # Extract documents from results
    documents = results.get('documents', [[]])[0]

    if not documents:
        # If no documents found, set message and redirect to index
        session['message'] = 'No results found.'
        return redirect(url_for('index'))

    # Render results template with retrieved documents
    return render_template('results.html', results=documents)

if __name__ == '__main__':
    # Run the Flask app in debug mode on port 8000
    app.run(debug=True, port=8000)
