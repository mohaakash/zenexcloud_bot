import json
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from dotenv import load_dotenv
from datetime import datetime
import threading

# new imports
import contextvars
import uuid
import copy

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load the knowledge base
with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

# Create vector store documents
documents = [
    Document(
        page_content=f"Question: {item.get('question','')}\nAnswer: {item.get('answer','')}",
        metadata={"question": item.get('question','')}
    ) for item in knowledge_base
]

# Embeddings & FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

# --- SESSION MANAGEMENT ---
# template for session data
TEMPLATE_SESSION_DATA = {
    "email": None,
    "phone": None,
    "problem_details": None,
    "escalation_step": 0,  # 0: initial, 1: asked for details, 2: asked for contact, 3: completed
    "chat_history": [],
    "awaiting_details": False,
    "awaiting_contact": False
}

# context var to track current session id (safe for async/threads)
_current_session = contextvars.ContextVar("current_session", default="default")

# store per-session ConversationBufferMemory and session_data
_session_memories = {}
_session_states = {}
_session_qa_chains = {}  # Store per-session QA chains
_lock = threading.RLock()  # Thread safety for session data

def _ensure_session(session_id: str):
    """Ensure memory, state, and QA chain exist for session_id."""
    with _lock:
        if session_id not in _session_memories:
            _session_memories[session_id] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
        if session_id not in _session_states:
            _session_states[session_id] = copy.deepcopy(TEMPLATE_SESSION_DATA)
        if session_id not in _session_qa_chains:
            # Create per-session QA chain with its own memory
            session_memory = _session_memories[session_id]
            _session_qa_chains[session_id] = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=session_memory,
                combine_docs_chain_kwargs={"prompt": chat_prompt}
            )

def set_current_session(session_id: str = None):
    """
    Set current session id for subsequent calls to get_response / save_session.
    Call this from your backend before calling get_response for a particular user.
    If not called, 'default' session is used.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    _current_session.set(session_id)
    _ensure_session(session_id)
    return session_id

def get_current_session_id():
    return _current_session.get()

def get_memory():
    sid = get_current_session_id()
    _ensure_session(sid)
    return _session_memories[sid]

def get_session_data():
    sid = get_current_session_id()
    _ensure_session(sid)
    return _session_states[sid]

def get_qa_chain():
    """Get the QA chain for current session"""
    sid = get_current_session_id()
    _ensure_session(sid)
    return _session_qa_chains[sid]

# Backward-compatible variable named `memory` used in existing functions:
# create a small proxy object so existing code referencing `memory.chat_memory` still works.
class _MemoryProxy:
    def __getattr__(self, name):
        return getattr(get_memory(), name)

memory = _MemoryProxy()

# Prompt
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful and polite customer support chatbot for ZenexCloud hosting services.

Rules:
- Reply professionally and clearly.
- Handle greetings (hi, hello, hey, etc.) with a friendly welcome message.
- If the query is in knowledge base, use the KB answer.
- For human support requests or issues not in KB, follow the escalation flow exactly.
- Always maintain context from previous messages in this conversation.
- Be concise but helpful in your responses.

Important: Follow the escalation steps in order and do NOT skip or repeat steps.
    """),
    ("human", "Context:\n{context}\n\nUser Question:\n{question}"),
    ("ai", "Answer:")
])

def is_greeting(user_input):
    """Check if user input is a greeting"""
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    return any(greeting in user_input.lower() for greeting in greetings)

def is_human_support_request(user_input):
    """Check if user is requesting human support"""
    support_keywords = ["human", "agent", "representative", "speak to someone", "talk to someone", 
                       "escalate", "supervisor", "manager", "help me", "support", "assistance"]
    return any(keyword in user_input.lower() for keyword in support_keywords)

def extract_contact_info(user_input):
    """Extract email and phone from user input"""
    sd = get_session_data()
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"\b(?:\+?\d{1,3})?[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    
    email_match = re.search(email_pattern, user_input)
    phone_match = re.search(phone_pattern, user_input)

    if email_match and not sd["email"]:  # Only update if not already set
        sd["email"] = email_match.group()
    if phone_match and not sd["phone"]:  # Only update if not already set
        sd["phone"] = phone_match.group()
    
    return bool(email_match or phone_match)

def has_no_details_response(user_input):
    """Check if user is indicating they have no details to provide"""
    no_details_phrases = ["no details", "don't have details", "not sure", "none", "no", 
                         "i don't know", "nothing specific", "can't think of any", "no info"]
    return any(phrase in user_input.lower() for phrase in no_details_phrases)

def extract_problem_details(user_input):
    """Extract problem details from user input"""
    sd = get_session_data()
    # Skip if it's just contact info or no details response
    if extract_contact_info(user_input) or has_no_details_response(user_input):
        return False
    
    # If input is longer than 15 chars and contains some meaningful content
    if len(user_input.strip()) > 15:
        if sd["problem_details"]:
            sd["problem_details"] += " " + user_input
        else:
            sd["problem_details"] = user_input
        return True
    return False

def handle_escalation_flow(user_input):
    """Handle the escalation flow logic"""
    sd = get_session_data()
    # Step 1: Check if we need to ask for details
    if sd["escalation_step"] == 0:
        sd["escalation_step"] = 1
        sd["awaiting_details"] = True
        return "Please provide us with more details about the problem you are facing so we can assist you more smoothly."
    
    # Step 2: User responded to details request
    elif sd["escalation_step"] == 1 and sd["awaiting_details"]:
        # Extract any problem details or check if they said no details
        extract_problem_details(user_input)
        has_no_details = has_no_details_response(user_input)
        
        # Move to next step regardless of whether they provided details or not
        sd["escalation_step"] = 2
        sd["awaiting_details"] = False
        sd["awaiting_contact"] = True
        
        if has_no_details:
            sd["problem_details"] = "No specific details provided"
        
        return "Please provide us with your email and phone number, our technical team will contact you shortly."
    
    # Step 3: User responded to contact request
    elif sd["escalation_step"] == 2 and sd["awaiting_contact"]:
        # Extract contact info
        contact_provided = extract_contact_info(user_input)
        
        if contact_provided:
            sd["escalation_step"] = 3
            sd["awaiting_contact"] = False
            return "We will contact you shortly. Thank you for being with ZenexCloud."
        else:
            # If no contact info found, ask again
            return "I didn't find any email or phone number in your message. Please provide your email and phone number so our technical team can contact you."
    
    # Step 4: Escalation completed
    elif sd["escalation_step"] == 3:
        return "Your request has been submitted to our technical team. Is there anything else I can help you with?"
    
    return None

def save_session():
    """Save session summary to JSON for current session"""
    try:
        sid = get_current_session_id()
        with _lock:
            if sid not in _session_memories:
                return  # Session doesn't exist
            
            mem = _session_memories[sid]
            sd = _session_states[sid]
            
        # Safely extract chat summary
        summary = ""
        if hasattr(mem, 'chat_memory') and hasattr(mem.chat_memory, 'messages'):
            summary = " ".join([msg.content for msg in mem.chat_memory.messages])
        
        data_to_save = {
            "session_id": sid,
            "timestamp": datetime.now().isoformat(),
            "email": sd["email"],
            "phone": sd["phone"],
            "problem_details": sd["problem_details"],
            "escalation_step": sd["escalation_step"],
            "chat_summary": summary
        }

        os.makedirs("sessions", exist_ok=True)
        filename = f"sessions/session_{sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Session saved to {filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save session: {str(e)}")

def get_response(user_query):
    """
    Main function to get response for any user query
    Args:
        user_query (str): User input/question
    Returns:
        str: Bot response
    """
    if not user_query or not user_query.strip():
        return "I didn't receive any input. Please ask me something!"
    
    user_input = user_query.strip()
    
    try:
        sd = get_session_data()
        mem = get_memory()

        # Handle greetings
        if is_greeting(user_input) and sd["escalation_step"] == 0:
            response = "Hello! Welcome to ZenexCloud customer support. How can I assist you today?"
            mem.chat_memory.add_user_message(user_input)
            mem.chat_memory.add_ai_message(response)
            return response
        
        # Check if we're in escalation flow
        if sd["escalation_step"] > 0:
            escalation_response = handle_escalation_flow(user_input)
            if escalation_response:
                mem.chat_memory.add_user_message(user_input)
                mem.chat_memory.add_ai_message(escalation_response)
                return escalation_response
        
        # Check if user wants human support
        if is_human_support_request(user_input) and sd["escalation_step"] == 0:
            escalation_response = handle_escalation_flow(user_input)
            mem.chat_memory.add_user_message(user_input)
            mem.chat_memory.add_ai_message(escalation_response)
            return escalation_response
        
        # Default: Use the knowledge base and LLM with per-session QA chain
        qa_chain = get_qa_chain()
        response = qa_chain.invoke({"question": user_input})
        answer = response["answer"]
        
        # The QA chain automatically handles memory, but we ensure consistency
        # by also updating our session memory if needed
        return answer
        
    except Exception as e:
        error_msg = "I apologize, but I'm having trouble processing your request. Please try again or contact our support team."
        print(f"[ERROR] {str(e)}")
        try:
            get_memory().chat_memory.add_ai_message(error_msg)
        except:
            pass  # Don't let memory errors compound the issue
        return error_msg

def cleanup_old_sessions(max_sessions=100):
    """Clean up old sessions to prevent memory leaks"""
    with _lock:
        if len(_session_memories) > max_sessions:
            # Remove oldest sessions (simple FIFO cleanup)
            sessions_to_remove = list(_session_memories.keys())[:-max_sessions//2]
            for session_id in sessions_to_remove:
                _session_memories.pop(session_id, None)
                _session_states.pop(session_id, None)
                _session_qa_chains.pop(session_id, None)
            print(f"[INFO] Cleaned up {len(sessions_to_remove)} old sessions")

def chatbot():
    """Main chatbot loop for command line interface"""
    # for CLI we create and set a local session id so memory works per run
    sid = set_current_session("cli_" + str(uuid.uuid4())[:8])
    print("Welcome to ZenexCloud Customer Support Chatbot! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            save_session()
            print("Chatbot: Goodbye!")
            break
        
        # Get response using the main function
        response = get_response(user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot()