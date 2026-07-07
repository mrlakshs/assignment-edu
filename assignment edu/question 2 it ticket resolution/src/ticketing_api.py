import logging

logging.basicConfig(level=logging.INFO)

class TicketResponder:
    def __init__(self, sender_email="it-support@enterprise.com"):
        """
        Simulates the integration with an enterprise ticketing system 
        (like ServiceNow or Zendesk).
        """
        self.sender_email = sender_email

    def generate_response(self, ticket_id: str, match_results: list, threshold: float) -> dict:
        """
        Evaluates the ML match results. If the confidence is high enough, 
        it formats an auto-reply. If not, it formats an escalation ticket.
        """
       
        top_match = match_results[0]
        confidence = top_match['similarity_score']
        
        if confidence >= threshold:
            
            response = {
                "status": "AUTO_RESOLVED",
                "ticket_id": ticket_id,
                "assigned_category": top_match['category'],
                "action": "EMAIL_USER",
                "email_body": (
                    f"Hello,\n\n"
                    f"Based on your ticket, we suggest the following standard solution:\n\n"
                    f"--- {top_match['solution']} ---\n\n"
                    f"Did this resolve your issue?"
                ),
                "confidence_score": round(confidence, 2)
            }
            logging.info(f"System Action: Auto-Reply generated for Ticket {ticket_id}")
            
        else:
            
            response = {
                "status": "ESCALATED",
                "ticket_id": ticket_id,
                "assigned_category": "Requires Manual Review",
                "action": "ROUTE_TO_AGENT",
                "email_body": "Your ticket has been received and routed to a Tier 2 support agent.",
                "confidence_score": round(confidence, 2)
            }
            logging.info(f"System Action: Ticket {ticket_id} escalated to human agent.")
            
        return response