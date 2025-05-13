import os
import litellm

if __name__ == "__main__":
    model = os.getenv("AGENT_SUPERVISOR_MODEL")
    prompt = "Explain what PMSSync is."
    print(f"[TEST] Calling LiteLLM with model: {model}")
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        print(f"[TEST] LiteLLM response: {response['choices'][0]['message']['content']}")
    except Exception as e:
        print(f"[TEST] LiteLLM call failed: {e}")