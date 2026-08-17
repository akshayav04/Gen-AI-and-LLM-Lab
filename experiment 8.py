from transformers import pipeline

# ----------------------------------------
# 1. Load a pretrained code-generation model
# ----------------------------------------

assistant = pipeline(
    task="text2text-generation",
    model="Salesforce/codet5-base"
)

# ----------------------------------------
# 2. Function to generate model response
# ----------------------------------------

def generate_response(prompt, max_tokens=250):
    result = assistant(
        prompt,
        max_new_tokens=max_tokens,
        do_sample=False
    )
    return result[0]["generated_text"]

# ----------------------------------------
# 3. Code Generation
# ----------------------------------------

code_task = """
Write a Python program to calculate the factorial of a number
using a recursive function.

Requirements:
1. Accept a number from the user.
2. Use recursion.
3. Display the factorial.
4. Handle negative input.
"""

generation_prompt = f"""
Generate Python code for the following task.

Task:
{code_task}

Return only a complete and properly indented Python program.
"""

generated_code = generate_response(generation_prompt)

# ----------------------------------------
# 4. Debugging Assistant
# ----------------------------------------

faulty_code = """
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(number)
    return average

values = [10, 20, 30, 40]
print("Average:", calculate_average(values))
"""

debugging_prompt = f"""
Analyze the following Python program.

Code:
{faulty_code}

Tasks:
1. Identify the error.
2. Explain the cause briefly.
3. Provide the corrected Python program.

Use the following format:

Error:
Explanation:
Corrected Code:
"""

debugging_result = generate_response(debugging_prompt)

# ----------------------------------------
# 5. Display Results
# ----------------------------------------

print("AI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT")
print("=" * 60)

print("\n1. GENERATED CODE")
print("-" * 60)
print(generated_code)

print("\n2. DEBUGGING RESULT")
print("-" * 60)
print(debugging_result)
