from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

import torch
from transformers import pipeline
# Load TinyLlama model
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                torch_dtype=torch.bfloat16, device_map="auto")
# Command to start the bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I am pikabot, your AI assistant. Ask me anything!")

# Handle incoming messages and generate responses
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    print(f"Received message: {user_message}")
    
    # Prepare the chat prompt
    messages = [
    {"role": "user", "content": user_message}
    ]
    # Format the input and generate the response
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95) 
    # Send the generated response to the user
    generated_text = outputs[0]["generated_text"]
    # Remove system context from the generated text (strip out the system message and tokens)
    clean_response = generated_text.replace('</s>', '').replace('<|user|>', '').replace('<|assistant|>', '').replace('<|system|>', '').replace(user_message,'').strip()
    print(f"Generated response: {clean_response}")
    # Send the cleaned response to the user
    await update.message.reply_text(clean_response)

def main():
    application = Application.builder().token("7945499297:AAEx4Nk-whfiKge778ScjOF9DHSAeMOIwMo").build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Start the bot
    print("Pika bot is running...")
    application.run_polling()

if __name__ == "__main__":
    main()
