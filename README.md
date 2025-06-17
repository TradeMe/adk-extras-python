# adk-extras-python
Open-source extensions to the python Google Agent Development Kit

This adds patches and extensions to [google-adk](https://github.com/google/adk-python). These will usually also be 
contributed upstream to the core library at the same time - and when adopted there this library will drop it's 
implementation and pass through to the upstream version (meaning users of this lib will end up importing the core 
versions instead) whenever doing so would be non-breaking.

## Patches

### LiteLLM Streaming

A drop in replacement for LiteLLM, but with streaming fixed:

```python
from google.adk.agents import Agent
from adk_extras.models.lite_llm import LiteLlmWithStreaming as LiteLlm

agent = Agent(
    name="weather_agent_gpt",
    model=LiteLlm(model="openai/gpt-4o"),
    description="Provides weather information using OpenAI's GPT.",
    instruction="You are a helpful weather assistant powered by GPT-4o. "
                "Use the 'get_weather' tool for city weather requests. "
                "Present information clearly.",
    tools=[get_weather],
)
...
```
