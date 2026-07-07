from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)

class TicketEmbedder:
    def __init__(self, model_name: str):
        """
        Loads the NLP transformer model. We use Sentence-BERT because 
        it understands the *meaning* of a sentence, ignoring typos and noise.
        """
        logging.info(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
    def get_embedding(self, text: str):
        """Converts text into a fixed-length numerical vector."""
        return self.model.encode([text])[0]
        
    def get_batch_embeddings(self, text_list: list):
        """Efficiently converts thousands of historical tickets to vectors."""
        return self.model.encode(text_list, show_progress_bar=True)