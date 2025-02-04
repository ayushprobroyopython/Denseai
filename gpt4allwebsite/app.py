from flask import Flask, render_template, request, jsonify
from gpt4all import GPT4All
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize GPT4All model (adjust path if needed)
model_path = "Llama-3.2-1B-Instruct-Q4_0.gguf" # Example path, replace with your actual model path
gpt_model = GPT4All(model_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Generate response from GPT4All
        response = gpt_model.generate(user_message)  # Adjust parameters as needed
        # Example settings
        # Extract the generated text (different models might have slightly different output structures)
        bot_response = response # For simple models.  More complex models might return a dictionary.
                               # If so, you'd extract the text like: bot_response = response['choices'][0]['text']
        

        return jsonify({"response": bot_response})

    except Exception as e:
        print(f"Error during generation: {e}")  # Important for debugging
        return jsonify({"error": "Error generating response"}), 500



if __name__ == "__main__":
    with gpt_model.chat_session():
        app.run(debug=False, port=5000, host='0.0.0.0')  # Set debug=False for production
