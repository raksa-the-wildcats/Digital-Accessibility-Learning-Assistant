import gradio as gr
import os
from config import Config
from utils.qa_chain import QAChain

class AccessibilityChatbot:
    def __init__(self):
        self.qa_chain = None
        self.is_initialized = False
    
    def initialize(self):
        """Initialize the chatbot and knowledge base."""
        try:
            Config.validate()
            self.qa_chain = QAChain()
            success = self.qa_chain.initialize_knowledge_base()
            if success:
                self.is_initialized = True
                return "✅ Chatbot initialized successfully!"
            else:
                return "❌ Failed to initialize knowledge base. Please check if PDFs are in the data/pdfs directory."
        except Exception as e:
            return f"❌ Initialization error: {str(e)}"
    
    def chat(self, message, history):
        """Handle chat interactions."""
        print("[DEBUG] chat called")
        print(f"[DEBUG] message: {message}")
        print(f"[DEBUG] history (in): {history}")
        if not self.is_initialized:
            init_result = self.initialize()
            if not self.is_initialized:
                result = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": init_result}
                ]
                print(f"[DEBUG] history (out): {result}")
                return result
        
        if not message.strip():
            result = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Please ask a question about web accessibility."}
            ]
            print(f"[DEBUG] history (out): {result}")
            return result
        
        try:
            # Check if qa_chain is available
            if not self.qa_chain:
                result = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": "❌ Chatbot not properly initialized. Please click 'Initialize' first."}
                ]
                print(f"[DEBUG] history (out): {result}")
                return result
            
            # Get answer from QA chain
            result_data = self.qa_chain.get_answer(message)
            
            # Format response
            response = result_data["answer"]
            if result_data["sources"]:
                response += f"\n\n**Sources:** {', '.join(result_data['sources'])}"
            
            result = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]
            print(f"[DEBUG] history (out): {result}")
            return result
            
        except ImportError as e:
            error_msg = f"❌ Missing dependency: {str(e)}. Please install required packages: pip install openai sentence-transformers"
            result = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            print(f"[DEBUG] history (out): {result}")
            return result
        except Exception as e:
            error_msg = f"❌ Error processing your message: {str(e)}"
            result = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": error_msg}
            ]
            print(f"[DEBUG] history (out): {result}")
            return result
    
    def clear_chat(self):
        """Clear chat history."""
        return []
    
    def get_example_questions(self):
        """Get example questions for the interface."""
        return [
            "What are the WCAG 2.1 guidelines for color contrast?",
            "How do I make images accessible?",
            "What is the proper way to use ARIA labels?",
            "How can I make forms more accessible?",
            "What are the requirements for keyboard navigation?"
        ]

def create_interface():
    """Create the Gradio interface."""
    chatbot_instance = AccessibilityChatbot()
    
    with gr.Blocks(title=Config.APP_TITLE) as demo:
        gr.Markdown(f"# {Config.APP_TITLE}")
        gr.Markdown(Config.APP_DESCRIPTION)
        
        # Status indicator
        status = gr.Textbox(
            label="Status",
            value="Click 'Initialize' to start the chatbot",
            interactive=False
        )
        
        # Initialize button
        init_btn = gr.Button("Initialize Chatbot", variant="primary")
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat History",
            height=400,
            show_label=True,
            type='messages'
        )
        
        # Input components
        with gr.Row():
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Ask a question about web accessibility...",
                scale=4
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Control buttons
        with gr.Row():
            clear_btn = gr.Button("Clear Chat", variant="secondary")
            refresh_btn = gr.Button("Refresh Knowledge Base", variant="secondary")
        
        # Example questions
        gr.Markdown("### Example Questions:")
        example_questions = chatbot_instance.get_example_questions()
        for question in example_questions:
            gr.Markdown(f"- {question}")
        
        # Setup event handlers
        def initialize_chatbot():
            result = chatbot_instance.initialize()
            return result
        
        def refresh_knowledge_base():
            if chatbot_instance.qa_chain:
                success = chatbot_instance.qa_chain.initialize_knowledge_base(force_rebuild=True)
                if success:
                    return "✅ Knowledge base refreshed successfully!"
                else:
                    return "❌ Failed to refresh knowledge base."
            return "❌ Chatbot not initialized."
        
        # Event bindings
        init_btn.click(
            initialize_chatbot,
            outputs=[status]
        )
        
        submit_btn.click(
            chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        msg.submit(
            chatbot_instance.chat,
            inputs=[msg, chatbot],
            outputs=[chatbot]
        ).then(
            lambda: "",
            outputs=[msg]
        )
        
        clear_btn.click(
            chatbot_instance.clear_chat,
            outputs=[chatbot]
        )
        
        refresh_btn.click(
            refresh_knowledge_base,
            outputs=[status]
        )
    
    return demo

if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("Warning: .env file not found. Please create one with your OPENAI_API_KEY")
        print("Example .env content:")
        print("OPENAI_API_KEY=your_api_key_here")
    
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )