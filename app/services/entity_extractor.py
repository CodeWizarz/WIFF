from openai import AsyncOpenAI
from typing import List, Dict, Any
import json
from app.config import settings

class EntityExtractor:
    """Extract entities from text using LLM"""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text using structured output.
        
        Returns:
            List of entities with type, name, and properties
        """
        prompt = f"""Extract key entities from the following text.
Focus on: People, Organizations, Concepts, Events, Technologies.

Return a JSON array of entities with this format:
[
  {{"type": "Person", "name": "John Doe", "properties": {{"role": "CEO"}}}},
  {{"type": "Organization", "name": "OpenAI", "properties": {{}}}},
  ...
]

Text:
{text[:2000]}  # Limit to avoid context overflow

Return only the JSON array, no additional text."""

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an entity extraction system. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # Handle both {"entities": [...]} and [...] formats
            if isinstance(result, dict) and "entities" in result:
                return result["entities"]
            elif isinstance(result, list):
                return result
            else:
                return []
        except json.JSONDecodeError:
            # Fallback: return empty if parsing fails
            return []
