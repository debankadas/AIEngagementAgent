import os
import json
from typing import Dict, List, Any
from datetime import datetime



# Initialize mock tools and storage
lead_database = {}
sessions = {}


class MockBusinessCardScanner:
    def scan(self):
        """Mock business card scanning"""
        return {
            "full_name": "Rohan Singh",
            "company": "Café Brew Haven",
            "role": "Owner",
            "email": "rohan@brewhavengroup.com",
            "phone": "+91 98765 43210",
            "website": "www.brewhavengroup.com",
            "location": "Pune, Maharashtra, India"
        }


def save_state(state, session_id):
    """Save the state to memory and optionally to disk"""
    sessions[session_id] = state
    os.makedirs("data", exist_ok=True)
    with open(f"data/session_{session_id}.json", "w") as f:
        json.dump(state, f, indent=2)


def load_state(session_id):
    """Load the state from memory or disk"""
    if session_id in sessions:
        return sessions[session_id]
    try:
        with open(f"data/session_{session_id}.json", "r") as f:
            state = json.load(f)
            sessions[session_id] = state
            return state
    except FileNotFoundError:
        return None


def initialize_state(session_id):
    """Initialize a new conversation state"""
    state = {
        "conversation_history": [],
        "lead_info": {},
        "follow_up_tasks": [],
        "current_stage": "greeting",
        "business_context": {
            "company_name": "ABC Roasters",
            "event_name": "CMPL Mumbai Expo 2025",
            "company_info": "Premium coffee supplier specializing in ethically sourced beans.",
            "products_info": "We offer various blends including Vanilla Blend and Cold Brew Blend."
        },
        "session_id": session_id
    }
    save_state(state, session_id)
    return state


def process_user_input(state, user_input):
    """Process user input and add to conversation history"""
    state["conversation_history"].append({
        "role": "human",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    })

    # Extract basic lead information from input
    extract_lead_info(state, user_input)

    # Generate assistant response
    generate_response(state)

    # Save the updated state
    save_state(state, state["session_id"])

    return state


def extract_lead_info(state, user_input):
    """Simple rule-based extraction of lead information"""
    lead_info = state.get("lead_info", {})

    # Look for name patterns
    if "my name is" in user_input.lower():
        name_part = user_input.lower().split("my name is")[1].strip().split()[0]
        lead_info["full_name"] = name_part.title()

    # Look for company mentions
    if "company" in user_input.lower() or "work for" in user_input.lower() or "own" in user_input.lower():
        for prefix in ["i work for", "i own", "my company is", "called", "named"]:
            if prefix in user_input.lower():
                company_part = user_input.lower().split(prefix)[1].strip().split()[0]
                lead_info["company"] = company_part.title()

    # Check for mentions of Rohan (from example scenario)
    if "rohan" in user_input.lower() and "singh" in user_input.lower():
        # This is our example lead - use complete information
        lead_info.update({
            "full_name": "Rohan Singh",
            "company": "Café Brew Haven",
            "role": "Owner",
            "location": "Pune"
        })

    # Look for product interests
    if any(word in user_input.lower() for word in ["coffee", "beans", "blend", "roast", "vanilla", "cold brew"]):
        products = lead_info.get("products_of_interest", [])

        for product in ["coffee", "beans", "vanilla blend", "cold brew"]:
            if product in user_input.lower() and product not in products:
                products.append(product)

        if products:
            lead_info["products_of_interest"] = products

    # Update state
    state["lead_info"] = lead_info


def generate_response(state):
    """Generate a response based on the current stage"""
    current_stage = state["current_stage"]

    # Stage-specific responses
    if current_stage == "greeting":
        response = "Hello! Welcome to ABC Roasters booth at CMPL Mumbai Expo. I'm Jane, your AI assistant. What brings you to our booth today?"
        state["current_stage"] = "info_gathering"

    elif current_stage == "info_gathering":
        response = "That's great! Could you tell me a bit about your company and what type of coffee products you're looking for? If you'd like, you can also show me your business card for quicker information capture."

        # Check if we have enough info to move to the next stage
        if "full_name" in state["lead_info"] and "company" in state["lead_info"]:
            state["current_stage"] = "business_card"

    elif current_stage == "business_card":
        # Simulate business card scanning
        if "I can show you my business card" in state["conversation_history"][-1]["content"]:
            # Scan the business card
            scanner = MockBusinessCardScanner()
            contact_info = scanner.scan()
            state["lead_info"].update(contact_info)

            response = f"Thank you, {contact_info['full_name']}! I've captured your details from Café Brew Haven. What's your preferred method for follow-up communication?"
        else:
            response = "Great! Based on your interests, we have several coffee blends that might be perfect for your needs. Would you be able to show me your business card so I can quickly capture your contact details?"

        # Advance to product discussion after business card or if we have enough information
        if "email" in state["lead_info"] or "phone" in state["lead_info"]:
            state["current_stage"] = "product_discussion"

    elif current_stage == "product_discussion":
        products = state["lead_info"].get("products_of_interest", [])
        if "vanilla blend" in products or "coffee" in products:
            response = "Our Vanilla Blend is a premium mix of Ethiopian and Colombian beans with subtle vanilla notes. It's perfect for espresso-based drinks. We also have a Cold Brew Blend that's specifically designed for cold brewing methods. What minimum order quantity would you be considering?"
        else:
            response = "We offer a range of premium coffee blends. Our most popular options include our Vanilla Blend and Cold Brew Blend. The Vanilla Blend has subtle vanilla notes and works excellently in espresso-based drinks. Are you interested in particular flavor profiles?"

        state["current_stage"] = "objection_handling"

    elif current_stage == "objection_handling":
        response = "That makes sense. For first-time orders, we offer 5kg sample packs so you can test the quality before committing to larger quantities. We can deliver within 5-7 days to locations in Pune. Do you have any concerns about pricing or quality consistency?"
        state["current_stage"] = "next_steps"

    elif current_stage == "next_steps":
        response = "I'd be happy to arrange for sample packs of both our Vanilla Blend and Cold Brew Blend to be sent to your cafés. We'll also send you a detailed pricing comparison. Would it be helpful to schedule a call with our account manager next week to discuss your specific requirements?"
        state["current_stage"] = "closing"

    elif current_stage == "closing":
        response = f"Thank you for visiting our booth, {state['lead_info'].get('full_name', 'there')}! To summarize, we'll send sample packs of our Vanilla Blend and Cold Brew Blend to your cafés, along with a pricing comparison. Our team will follow up within 3 business days. Enjoy the rest of the expo!"
        state["current_stage"] = "completed"

        # Generate follow-ups when completing
        generate_follow_ups(state)

    else:  # completed or unknown
        response = "Thank you for your interest in ABC Roasters! Our team will be in touch soon with the information we discussed."

    # Add response to conversation history
    state["conversation_history"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })

    return response


def generate_follow_ups(state):
    """Generate follow-up tasks"""
    if not state["lead_info"]:
        return

    # Email follow-up
    email_task = {
        "type": "email",
        "status": "pending",
        "recipient": state["lead_info"].get("email", ""),
        "recipient_name": state["lead_info"].get("full_name", ""),
        "subject": f"Follow-up from ABC Roasters at {state['business_context']['event_name']}",
        "content": f"Dear {state['lead_info'].get('full_name', '')},\n\nThank you for visiting our booth at {state['business_context']['event_name']}. As discussed, we'll be sending sample packs of our coffee blends for your cafés.\n\nLooking forward to working with you.\n\nBest regards,\nABC Roasters Team",
        "created_at": datetime.now().isoformat()
    }

    # Meeting follow-up
    meeting_task = {
        "type": "meeting",
        "status": "pending",
        "contact_name": state["lead_info"].get("full_name", ""),
        "contact_email": state["lead_info"].get("email", ""),
        "details": f"Follow-up call to discuss coffee blend requirements for {state['lead_info'].get('company', '')}",
        "due_date": "Next week",
        "created_at": datetime.now().isoformat()
    }

    # Sample sending task
    sample_task = {
        "type": "task",
        "status": "pending",
        "details": f"Send Vanilla Blend and Cold Brew Blend samples to {state['lead_info'].get('full_name', '')} at {state['lead_info'].get('company', '')}",
        "due_date": "Within 3 days",
        "created_at": datetime.now().isoformat()
    }

    state["follow_up_tasks"] = [email_task, meeting_task, sample_task]

    # Save lead to database
    lead_id = f"lead_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    lead_database[lead_id] = {
        "id": lead_id,
        "lead_info": state["lead_info"],
        "follow_up_tasks": state["follow_up_tasks"],
        "source_event": state["business_context"]["event_name"],
        "created_at": datetime.now().isoformat()
    }

    print(f"Lead saved to database with ID: {lead_id}")


def run_demo():
    """Run a demonstration conversation"""
    print("Starting Trade Show Lead Management Demo")
    print("=" * 50)

    # Initialize a new session
    session_id = f"demo_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    state = initialize_state(session_id)

    # Define demo conversation
    demo_messages = [
        "Hello, I'm interested in learning about your coffee products.",
        "I'm Rohan Singh, I own multiple cafes in Pune.",
        "I can show you my business card.",
        "WhatsApp works best for me.",
        "I'm interested in your Vanilla Blend and Cold Brew options.",
        "I'd like to start with a small order to test the quality.",
        "Yes, pricing is important as my cafe customers are value-conscious.",
        "That sounds good. I would appreciate the samples.",
        "Yes, a call next week would be helpful.",
        "Thank you for the information!"
    ]

    # Process each message
    for i, message in enumerate(demo_messages):
        print(f"\nUser: {message}")
        state = process_user_input(state, message)
        last_response = state["conversation_history"][-1]["content"]
        print(f"AI: {last_response}")
        print(f"Current stage: {state['current_stage']}")

    # Print final state
    print("\n" + "=" * 50)
    print("Conversation completed")
    print("\nLead information collected:")
    print(json.dumps(state["lead_info"], indent=2))

    print("\nFollow-up tasks:")
    for task in state["follow_up_tasks"]:
        print(f"- {task['type']}: {task.get('details', task.get('subject', ''))}")

    return state


if __name__ == "__main__":
    run_demo()