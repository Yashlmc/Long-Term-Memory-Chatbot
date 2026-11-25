import json
import re

from llm.openrouter_client import OpenRouterClient


ALLOWED_MEMORY_TYPES = {
    "profile",
    "preference",
    "interest",
    "skill",
    "project",
    "goal",
    "habit",
    "other",
}

MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction system for a general-purpose chatbot.

Identify information from the user's message that is useful to
remember across future conversations.

Remember:
- Stable personal facts
- User preferences
- Long-term interests
- Skills
- Ongoing projects
- Long-term goals
- Recurring habits

Do NOT remember:
- Temporary requests
- Greetings
- Casual conversation
- Questions without personal information
- Temporary states such as "I am hungry right now"
- Information about unrelated third parties
- Passwords, API keys, bank details, or sensitive medical diagnoses

Return ONLY a valid JSON object with no extra prose.

Required format:
{
  "should_remember": true,
  "memories": [
    {
      "text": "concise third-person memory (e.g., User prefers Python over Java)",
      "type": "preference",
      "importance": 0.8,
      "confidence": 1.0
    }
  ]
}

Allowed types:
profile, preference, interest, skill, project, goal, habit, other

Importance range: [0.0, 1.0]
Confidence range: [0.0, 1.0]

If there is nothing worth remembering:
{
  "should_remember": false,
  "memories": []
}
"""


class MemoryExtractor:
    def __init__(self, llm=None):
        self.llm = llm or OpenRouterClient()

    def _extract_json(self, response):
        if not isinstance(response, str):
            return None

        response = response.strip()

        # Case 1: Already valid JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Case 2: JSON wrapped in ```json ... ```
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Case 3: First JSON object span
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start : end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _validate_result(self, result):
        if not isinstance(result, dict):
            return None
        if not isinstance(result.get("should_remember"), bool):
            return None
        if not isinstance(result.get("memories"), list):
            return None

        cleaned_memories = []
        for memory in result["memories"]:
            if not isinstance(memory, dict):
                return None

            required_fields = ("text", "type", "importance", "confidence")
            if not all(field in memory for field in required_fields):
                return None

            text = memory["text"]
            mem_type = str(memory["type"]).strip().lower()
            importance = memory["importance"]
            confidence = memory["confidence"]

            if not isinstance(text, str) or not text.strip():
                return None

            if mem_type not in ALLOWED_MEMORY_TYPES:
                mem_type = "other"

            # Explicitly reject booleans (bool is a subclass of int in Python)
            if (
                isinstance(importance, bool)
                or not isinstance(importance, (int, float))
                or not 0 <= importance <= 1
            ):
                return None

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                return None

            cleaned_memories.append(
                {
                    "text": text.strip(),
                    "type": mem_type,
                    "importance": float(importance),
                    "confidence": float(confidence),
                }
            )

        return {
            "should_remember": result["should_remember"] and bool(cleaned_memories),
            "memories": cleaned_memories,
        }

    def extract(self, user_message, retries=1):
        messages = [
            {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": user_message},
        ]

        last_raw = ""
        for _ in range(retries + 1):
            last_raw = self.llm.generate_response(messages, temperature=0.0)
            parsed = self._extract_json(last_raw)
            validated = self._validate_result(parsed)
            if validated is not None:
                return validated

        print("\nWarning: Memory extractor produced an invalid response.")
        print("Raw response:")
        print(last_raw)
        return {"should_remember": False, "memories": []}
