import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, AutoModelForCausalLM, T5Tokenizer
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

st.set_page_config(
    page_title="Cancer Information Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def is_repetitive_response(text, threshold=5):
    """Check if response contains excessive repetition"""
    words = text.lower().split()
    if len(words) < 10:
        return False
    
    
    word_counts = {}
    for word in words:
        if len(word) > 3:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    
    max_count = max(word_counts.values()) if word_counts else 0
    return max_count >= threshold

def contains_inappropriate_content(text):
    """Check for inappropriate content in medical context"""
    inappropriate_patterns = [
        r'\bmake money\b',
        r'\bmoney\b.*\bmoney\b.*\bmoney\b', 
        r'\bget rich\b',
        r'\bscam\b',
        r'\bcrypto\b'
    ]
    
    text_lower = text.lower()
    for pattern in inappropriate_patterns:
        if re.search(pattern, text_lower):
            return True
    return False

def clean_and_validate_response(response, max_length=500):
    """Clean and validate model response"""
    if not response or len(response.strip()) < 5:
        return None
    
    
    response = re.sub(r'\s+', ' ', response.strip())
    
    
    if len(response) > max_length:
        response = response[:max_length] + "..."
    
    
    if is_repetitive_response(response) or contains_inappropriate_content(response):
        return None
    
    return response

def get_fallback_response(intent, question):
    """Provide appropriate fallback responses"""
    cancer_info_responses = {
        "question": [
            "I understand you're looking for information about cancer. While I'm having technical difficulties with my main response system, I'd recommend consulting with healthcare professionals or reputable medical sources like the American Cancer Society or National Cancer Institute for accurate information.",
            "For reliable cancer information, please consult with your healthcare provider or visit trusted medical websites. I'm currently experiencing some technical issues with my response generation.",
            "I apologize, but I'm having trouble generating a proper response right now. For cancer-related questions, please speak with a medical professional or check resources like cancer.gov."
        ],
        "greeting": [
            "Hello! I'm here to help with cancer-related questions, though I'm currently experiencing some technical issues. How can I assist you today?",
            "Hi there! While I work through some technical difficulties, I'm still here to try to help with your cancer-related questions."
        ],
        "emotional": [
            "I understand this can be a difficult topic. While I'm having some technical issues, please remember that support is available through healthcare providers, counselors, and cancer support organizations.",
            "This is certainly challenging, and I wish I could provide better assistance right now. Please consider reaching out to cancer support groups or healthcare professionals for guidance."
        ]
    }
    
    
    question_lower = question.lower()
    if any(word in question_lower for word in ['hello', 'hi', 'hey']):
        intent = "greeting"
    elif any(word in question_lower for word in ['scared', 'worried', 'afraid', 'emotional', 'feel']):
        intent = "emotional"
    else:
        intent = "question"
    
    responses = cancer_info_responses.get(intent, cancer_info_responses["question"])
    return responses[0] 

@st.cache_resource
def load_models():
    """
    Loads all fine-tuned models and tokenizers from the Hugging Face Hub.
    """
    intent_repo = "Tobivictor/intent-classifier-distilbert"
    t5_qa_repo = "Tobivictor/t5-finetuned-cancer-qa"
    dialogpt_repo = "Tobivictor/dialogpt-finetuned-cancer"

    
    loading_placeholder = st.sidebar.empty()
    
    try:
        
        loading_placeholder.info("🔄 Loading AI models...")
        intent_tokenizer = AutoTokenizer.from_pretrained(intent_repo)
        intent_model = AutoModelForSequenceClassification.from_pretrained(intent_repo)

        
        try:
            t5_tokenizer = T5Tokenizer.from_pretrained(t5_qa_repo)
        except Exception:
            try:
                t5_tokenizer = AutoTokenizer.from_pretrained(
                    t5_qa_repo, 
                    trust_remote_code=True,
                    use_fast=False
                )
            except Exception:
                t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        
        t5_model = T5ForConditionalGeneration.from_pretrained(t5_qa_repo)
        
        
        dialogpt_tokenizer = AutoTokenizer.from_pretrained(dialogpt_repo)
        
        if dialogpt_tokenizer.pad_token is None:
            dialogpt_tokenizer.pad_token = dialogpt_tokenizer.eos_token
            
        dialogpt_model = AutoModelForCausalLM.from_pretrained(dialogpt_repo)

        loading_placeholder.success("✅ AI Models ready!")
        
        models = {
            "intent": (intent_tokenizer, intent_model),
            "t5": (t5_tokenizer, t5_model),
            "dialogpt": (dialogpt_tokenizer, dialogpt_model)
        }
        return models
    
    except Exception as e:
        loading_placeholder.error("❌ Failed to load AI models")
        with st.expander("🔍 Error Details", expanded=False):
            st.error(f"Error loading models: {str(e)}")
            st.info("Possible solutions:")
            st.info("• Check your internet connection")
            st.info("• Verify the Hugging Face model repositories exist")
            st.info("• Try: pip install --upgrade transformers sentencepiece")
        return None

@st.cache_resource
def setup_knowledge_base():
    """
    Sets up the knowledge base silently without frontend notifications.
    """
    try:
        df = pd.read_csv('cancer_comments_annotated.csv').dropna(subset=['cleaned_comment'])
        knowledge_base = df['cleaned_comment'].tolist()
        
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = embedding_model.encode(knowledge_base, convert_to_tensor=True)
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.cpu().detach().numpy())
        
        return knowledge_base, embedding_model, index

    except (FileNotFoundError, Exception):
        return None, None, None

with st.spinner("🚀 Initializing Cancer Information Chatbot..."):
    try:
        models = load_models()
        if models is None:
            st.stop()
            
        knowledge_base, embedding_model, faiss_index = setup_knowledge_base()
        
    except Exception as e:
        st.error("Failed to initialize the chatbot. Please refresh the page.")
        with st.expander("🔍 Technical Details", expanded=False):
            st.error(f"Error: {str(e)}")
        st.stop()

# --- 3. Helper Functions for Generation ---
def retrieve_context(query, top_k=3):
    """Retrieves the most relevant context from the knowledge base."""
    if faiss_index is None or knowledge_base is None:
        return ""
    
    try:
        query_embedding = embedding_model.encode([query])
        distances, indices = faiss_index.search(query_embedding, top_k)
        context = "\n".join([knowledge_base[i] for i in indices[0] if i < len(knowledge_base)])
        return context
    except Exception:
        return ""

def get_intent(text):
    """Predicts the intent of the user's input text."""
    try:
        tokenizer, model = models["intent"]
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        return model.config.id2label[predicted_class_id]
    except Exception:
        return "question"

def generate_t5_response(question):
    """Generates a response using the T5 Q&A model without context."""
    try:
        tokenizer, model = models["t5"]
        
        input_text = f"answer the question: {question}"
        
        inputs = tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_length=150,
                min_length=10,
                num_beams=3,
                early_stopping=True,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        
        cleaned_response = clean_and_validate_response(response)
        
        if cleaned_response is None:
            return get_fallback_response("question", question)
        
        return cleaned_response
        
    except Exception as e:
        st.error(f"T5 Generation Error: {str(e)}")
        return get_fallback_response("question", question)

def generate_dialogpt_response(prompt):
    """Generates a conversational response using DialoGPT with knowledge base context."""
    try:
        tokenizer, model = models["dialogpt"]
        
       
        context = retrieve_context(prompt)
        
       
        if context.strip():
            enhanced_prompt = f"Context: {context[:200]}...\n\nUser: {prompt}\nBot:"
        else:
            enhanced_prompt = f"User: {prompt}\nBot:"
        
        new_user_input_ids = tokenizer.encode(
            enhanced_prompt + tokenizer.eos_token, 
            return_tensors='pt',
            max_length=400,
            truncation=True
        )
        
        with torch.no_grad():
            chat_history_ids = model.generate(
                new_user_input_ids, 
                max_length=min(500, new_user_input_ids.shape[-1] + 100),
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                num_return_sequences=1
            )
        
        response = tokenizer.decode(
            chat_history_ids[:, new_user_input_ids.shape[-1]:][0], 
            skip_special_tokens=True
        )
        
        
        cleaned_response = clean_and_validate_response(response.strip())
        
        if cleaned_response is None:
            return get_fallback_response("emotional", prompt)
        
        return cleaned_response
        
    except Exception as e:
        st.error(f"DialoGPT Generation Error: {str(e)}")
        return get_fallback_response("emotional", prompt)

st.title("Cancer Information Chatbot 🤖")
st.write("This app uses multiple AI models to understand and respond to you. Choose a generation model from the sidebar")

st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose the generative model:",
    ("T5 Q&A Model", "DialoGPT Conversational Model")
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Information:**")
if selected_model == "T5 Q&A Model":
    st.sidebar.info("T5 model fine-tuned for cancer-related Q&A. Best for factual questions.")
else:
    st.sidebar.info("DialoGPT model fine-tuned for cancer conversations. Best for dialogue.")

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Quality Notice**: Responses are automatically filtered for quality and appropriateness. If you see generic responses, the model may be having issues.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question or share a thought..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("The AI is thinking..."):
            try:
                intent = get_intent(prompt)
                
                if 'question' in intent.lower():
                    if selected_model == "T5 Q&A Model":
                        response = generate_t5_response(prompt)
                    else:
                        response = generate_dialogpt_response(prompt)
                else:
                    if selected_model == "DialoGPT Conversational Model":
                        response = generate_dialogpt_response(prompt)
                    else:
                        response = "Thank you for sharing. I am primarily a Q&A model, but please feel free to ask anything about cancer-related topics."

                
                st.markdown(response)
                
            except Exception as e:
                response = "I apologize, but I encountered an error. Please try again or rephrase your question."
                st.error(f"Error: {str(e)}")
                st.markdown(response)

    
    st.session_state.messages.append({"role": "assistant", "content": response})

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

with st.sidebar.expander("🔧 Debug Info", expanded=False):
    st.write("Model loaded:", models is not None)
    st.write("System status: Ready")