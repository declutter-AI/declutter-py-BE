from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class TaskExtractionService:
    @staticmethod
    async def extract_tasks(context: str) -> List[str]:
        """
        Extract actionable tasks from user context using OpenAI's GPT model.
        
        Args:
            context (str): User's input text containing potential tasks
            
        Returns:
            List[str]: List of extracted actionable tasks
        """
        try:
            response =  client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a task extraction assistant. Extract clear, actionable tasks from the given context. Return only the tasks, nothing else."
                    },
                    {
                        "role": "user",
                        "content": f"Extract actionable tasks from this text: {context}"
                    }
                ],
                temperature=0.7,
            )
            
            # Extract the content from the response
            tasks_text = response.choices[0].message.content
            
            # Split the response into individual tasks and clean them up
            tasks = [
                task.strip().strip('- ')  # Remove leading/trailing spaces and bullet points
                for task in tasks_text.split('\n')
                if task.strip()  # Remove empty lines
            ]
            
            return tasks
            
        except Exception as e:
            raise Exception(f"Error extracting tasks: {str(e)}")
