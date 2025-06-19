# 🧠 Multi-Agent Research Assistant with LangGraph (build from scratch)

This project demonstrates a simple multi-agent system using [LangGraph](https://www.langgraph.dev/) and OpenAI's GPT-4o-mini model. It includes three specialized agents:

- **Research Agent** – searches for information using Tavily.
- **Analysis Agent** – analyzes and summarizes the research results.
- **Translation Agent** – translates the final analysis into the requested language.

All agents are orchestrated by a **Supervisor Agent** which assigns tasks in a sequential pipeline.

## 🚀 How It Works

1. **User input** a task that includes research, analysis, and translation.
2. **Supervisor agent** delegates the task step by step:
   - Research → Analysis → Translation
3. Each agent performs its role and passes the result to the next.

## 📦 Requirements

Install dependencies:

```bash
pip install -U langgraph langgraph-supervisor langchain-tavily "langchain[openai]"
```

You’ll need an `.env` file with:

```env
OPENAI_API_KEY=your-openai-key
TAVILY_API_KEY=your-tavily-key
```

## ▶️ Run the App

```bash
python main.py
```

> The flow diagram `graph.png` will be generated and opened automatically.

## 🛠 Tools Used

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Tavily Search API](https://www.tavily.com/)
- OpenAI GPT-4o-mini model
- Python 3.10+

## 🖼 Example Prompt

```text
Please find recent applications of AI in education. Then analyze the main benefits. Finally, translate your analysis into Chinese.
```
