import pandas as pd
from src.config_loader import load_config
from src.ocr_engine import extract_text_from_image
from src.nlp_pipeline import TicketEmbedder
from src.vector_db import TicketKnowledgeBase
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    config = load_config("configs/config.yaml")
    
   
    embedder = TicketEmbedder(config['models']['embedding_model'])
    vector_db = TicketKnowledgeBase(config['faiss']['embedding_dimension'])
    
   
    logging.info("Loading SQL Knowledge Base...")
   
    historical_data = pd.read_csv(config['paths']['knowledge_base'])
    

    embeddings = embedder.get_batch_embeddings(historical_data['raw_text'].tolist())
    vector_db.build_index(embeddings, historical_data)
    
   
    
    new_ticket_text = "i cant log into my laptop i lost the passcode" 
    logging.info(f"Processing new ticket: '{new_ticket_text}'")
    
    
    query_vector = embedder.get_embedding(new_ticket_text)
    matches = vector_db.search(query_vector, top_k=config['faiss']['top_k_results'])
    
    top_match = matches[0]
    print(f"\n---> DEBUG: The AI scored the similarity at: {top_match['similarity_score']:.2f}\n")
    
    if top_match['similarity_score'] >= config['thresholds']['confidence_score']:
        logging.info("\n--- AUTO-RESOLUTION TRIGGERED ---")
        logging.info(f"Assigned Category: {top_match['category']}")
        logging.info(f"Suggested Solution: {top_match['solution']}")
        logging.info(f"Confidence: {top_match['similarity_score']:.2f}")
    else:
        logging.info("\n--- ESCALATED TO HUMAN AGENT ---")
        logging.info("Confidence too low for auto-resolution.")

if __name__ == "__main__":
    main()