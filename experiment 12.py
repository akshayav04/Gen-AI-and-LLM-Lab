import os
import re
import time
import getpass
import gradio as gr
from groq import Groq

# ============================================================
# GROQ API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = getpass.getpass("Enter your Groq API key: ")

client = Groq(api_key=api_key)


# ============================================================
# RELEVANCE SCORE
# ============================================================

def calculate_relevance(prompt, response):

    stop_words = {
        "the", "a", "an", "is", "are", "was", "were",
        "to", "of", "in", "on", "for", "and", "or",
        "with", "what", "how", "why", "write", "explain",
        "describe", "give", "about"
    }

    prompt_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            prompt.lower()
        )
    )

    response_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            response.lower()
        )
    )

    important_words = prompt_words - stop_words

    if not important_words:
        return 100.0

    matched_words = important_words.intersection(
        response_words
    )

    score = (
        len(matched_words) /
        len(important_words)
    ) * 100

    return round(score, 2)


# ============================================================
# GENERATE AND EVALUATE
# ============================================================

def generate_and_evaluate(
    prompt,
    temperature,
    max_tokens
):

    if not prompt or not prompt.strip():
        return (
            "Please enter a valid prompt.",
            {"Status": "No prompt provided"}
        )

    try:

        start_time = time.perf_counter()

        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Generative AI assistant. "
                        "Provide accurate, clear and well-structured answers."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=float(temperature),
            max_tokens=int(max_tokens)
        )

        end_time = time.perf_counter()

        generated_response = (
            completion.choices[0]
            .message.content
            .strip()
        )

        latency = end_time - start_time

        word_count = len(
            generated_response.split()
        )

        character_count = len(
            generated_response
        )

        relevance_score = calculate_relevance(
            prompt,
            generated_response
        )

        evaluation = {
            "Model": "llama-3.1-8b-instant",
            "Response Time (seconds)": round(latency, 3),
            "Generated Word Count": word_count,
            "Generated Character Count": character_count,
            "Keyword Relevance Score (%)": relevance_score,
            "Temperature": float(temperature),
            "Maximum Tokens": int(max_tokens),
            "Status": "Successfully generated"
        }

        return generated_response, evaluation

    except Exception as error:

        return (
            "The application could not generate a response.",
            {
                "Status": "Error",
                "Error Message": str(error)
            }
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks() as application:

    gr.Markdown(
        """
        # Cloud-Based Generative AI Application

        Enter a prompt to generate content and evaluate
        the response produced by the cloud-hosted language model.
        """
    )

    with gr.Row():

        with gr.Column():

            prompt_input = gr.Textbox(
                label="Enter Prompt",
                placeholder=(
                    "Example: Explain the applications "
                    "of Generative AI in education."
                ),
                lines=6
            )

            temperature_input = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.1,
                label="Temperature"
            )

            max_tokens_input = gr.Slider(
                minimum=50,
                maximum=500,
                value=250,
                step=50,
                label="Maximum Tokens"
            )

            generate_button = gr.Button(
                "Generate and Evaluate"
            )

            clear_button = gr.ClearButton(
                [prompt_input]
            )

        with gr.Column():

            response_output = gr.Textbox(
                label="Generated Response",
                lines=14
            )

  E          evaluation_output = gr.JSON(
                label="Evaluation Metrics"
            )

    generate_button.click(
        fn=generate_and_evaluate,
        inputs=[
            prompt_input,
            temperature_input,
            max_tokens_input
        ],
        outputs=[
            response_output,
            evaluation_output
        ]
    )


# ============================================================
# LAUNCH
# ============================================================

print("Starting Gradio application...")

application.launch(
    share=True,
    debug=True
)
