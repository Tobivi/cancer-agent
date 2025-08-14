# Cancer Information Chatbot 🤖

A comprehensive AI-powered chatbot designed to provide cancer-related information and support through conversational interactions. The application leverages multiple fine-tuned models to deliver contextually relevant responses.

## 🌟 Features

- **Dual Model Architecture**: Choose between T5 Q&A model for factual questions or DialoGPT for conversational responses
- **Intent Classification**: Automatically detects user intent to route queries to the most appropriate model
- **Quality Control**: Built-in response validation and filtering for medical appropriateness
- **Interactive Chat Interface**: Clean, user-friendly Streamlit interface
- **Context-Aware Responses**: Enhanced responses using relevant background knowledge
- **Real-time Processing**: Fast response generation with optimized model parameters

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - DistilBERT for intent classification
  - T5 for question-answering
  - DialoGPT for conversational responses
- **Knowledge Base**: FAISS vector database with sentence transformers
- **Backend**: PyTorch, Transformers (Hugging Face)

## 📋 Prerequisites

- Python 3.8 or higher
- 4-8GB RAM recommended
- Internet connection for initial model downloads

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cancer-information-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the knowledge base** (optional)
   - Place `cancer_comments_annotated.csv` in the root directory
   - The file should contain a `cleaned_comment` column with cancer-related text data
   - The app will work without this file, but responses may be less comprehensive

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run cancer-agent.py
   ```

2. **Access the application**
   - Open your web browser and go to `http://localhost:8501`
   - The app will automatically download and load the required AI models on first run

## 📖 Usage

### Model Selection
- **T5 Q&A Model**: Best for direct, factual questions about cancer
- **DialoGPT Conversational Model**: Better for ongoing conversations and emotional support

### Interaction Tips
- Ask clear, specific questions for best results
- Use natural language - no special formatting required
- The chatbot can handle both medical questions and emotional support requests

### Example Queries
- "What are the early symptoms of breast cancer?"
- "I'm feeling scared about my diagnosis"
- "How does chemotherapy work?"
- "What should I expect during treatment?"

## 🏗️ Project Structure

```
cancer-information-chatbot/
├── cancer-agent.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── cancer_comments_annotated.csv   # Knowledge base (optional)
└── .gitignore                     # Git ignore file
```

## 🤖 Model Information

The application uses three fine-tuned models from Hugging Face:

1. **Intent Classifier**: `Tobivictor/intent-classifier-distilbert`
   - Classifies user input to determine appropriate response strategy

2. **T5 Q&A Model**: `Tobivictor/t5-finetuned-cancer-qa`
   - Generates factual answers to cancer-related questions

3. **DialoGPT Conversational**: `Tobivictor/dialogpt-finetuned-cancer`
   - Provides conversational responses with knowledge base integration

## ⚙️ Configuration

### Model Parameters
- Response length: 150-500 tokens
- Temperature: 0.8 (for response diversity)
- Repetition penalty: 1.2-1.3
- Quality filtering enabled

### Knowledge Base
- Uses sentence-transformers for embedding generation
- FAISS indexing for fast similarity search
- Top-3 context retrieval for enhanced responses

## 🔧 Troubleshooting

### Common Issues

**Models not loading:**
- Check internet connection
- Verify Hugging Face model repositories are accessible
- Try: `pip install --upgrade transformers`

**Out of memory errors:**
- Close other applications to free RAM
- Consider using smaller batch sizes
- Use CPU-only mode if necessary

**CSV file not found:**
- The app will work without the knowledge base file
- Ensure `cancer_comments_annotated.csv` is in the root directory
- Check that the CSV has a `cleaned_comment` column

## 📊 Performance Optimization

- Models are cached using `@st.cache_resource` for faster subsequent loads
- Response validation prevents inappropriate or repetitive outputs
- Optimized token limits for faster generation while maintaining quality

## ⚠️ Important Disclaimers

- **Medical Advice**: This chatbot is for informational purposes only and should not replace professional medical advice
- **Accuracy**: While trained on cancer-related data, responses should be verified with healthcare professionals
- **Emergency**: For medical emergencies, contact your healthcare provider or emergency services immediately

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:
- Check the troubleshooting section above
- Review the Streamlit documentation
- Open an issue in the project repository

## 🙏 Acknowledgments

- Hugging Face for providing the transformer models and infrastructure
- Streamlit for the excellent web app framework
- The medical and AI communities for advancing cancer care technology

---

**Version**: 1.0.0  
**Last Updated**: August 2025  
**Author**: [Oluwatobi oduyebo]
