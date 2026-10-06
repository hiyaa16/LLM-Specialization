import aisuite as ai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Create AISuite client
client = ai.Client()

# Tool function
def get_current_time():
    """
    Returns the current time as a string.
    """
    return datetime.now().strftime("%H:%M:%S")


prompt = "What time is it?"

response = client.chat.completions.create(
    model="groq:llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    tools=[get_current_time],
    max_turns=5
)

print(response.choices[0].message.content)