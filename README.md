# LLM Practical Portfolio – PUSL3205

**Module:** PUSL3205 – Large Language Models and Generative AI  
**Student Name:** HWS Sathsarani  
**Student ID:** 10967109

This repository documents the practical laboratory activities completed during the module, from traditional NLP to prompt engineering, building a Transformer from scratch, fine-tuning with LoRA, Retrieval-Augmented Generation (RAG) and AI agents.

## Laboratory Activities

| Lab | Title | Key Tools |
|---|---|---|
| [Lab 01](Lab-01-Machine-Learning/) | Sentiment Analysis Using Traditional Machine Learning | Python, NLTK, Scikit-learn |
| [Lab 02](Lab-02-LLM-Everyday-Life/) | AI in Everyday Challenges – *StillHour* product design | LLM as a product-development partner |
| [Lab 03](Lab-03-Tiny-Transformer/) | Developing a Tiny Transformer Language Model | PyTorch, TinyStories, Google Colab, Gradio |
| [Lab 04](Lab-04-Fine-Tune/) | Fine-Tuning an LLM Using LoRA | Hugging Face, PEFT, TRL, Qwen2.5-0.5B |
| [Lab 05](Lab-05-RAG-Pipeline/) | RAG Pipeline – University Knowledge Assistant | LangChain, Ollama, ChromaDB, Streamlit |
| [Lab 06](Lab-06-AI-Agents/) | Building AI Agents with Tools, Memory, RAG and Google Calendar (Parts A–D) | LangChain, Ollama, SQLite, Google Calendar API |

## Repository Structure

```
LLM-Practical-Portfolio/
├── README.md
├── Final-Reflection.md
├── Lab-01-Machine-Learning/      README, notebook/, screenshots/
├── Lab-02-LLM-Everyday-Life/     README, activity-2/, screenshots/, prototype/
├── Lab-03-Tiny-Transformer/      README, notebooks/, screenshots/
├── Lab-04-Fine-Tune/             README, notebooks/, screenshots/
├── Lab-05-RAG-Pipeline/          README, Python files, requirements.txt, screenshots/, group-activity/
└── Lab-06-AI-Agents/             README, Python files, requirements.txt, screenshots/
```

Each lab folder contains a `README.md` with the objective, tasks and implementation, results and evidence, observations and learning outcomes. Credentials, tokens, databases, model files and university documents are excluded with `.gitignore`.

## Final Reflection
See [Final-Reflection.md](Final-Reflection.md).

## Use of Generative AI
This assessment is in the **Assisted** category. Claude (Anthropic) was used as an assistant throughout the labs. All code was run, and all experiments, results and screenshots were produced, by the student on their own computer or in Google Colab. Every AI suggestion was reviewed and tested before use.

| Area | How AI was used |
|---|---|
| Lab 01 | Explaining concepts, fixing the Anaconda installation, and suggesting notebook section headings |
| Lab 02 | AI helped improve the prompts and create the visual prototype board |
| Lab 03 | Debugging errors in the notebook and explaining the training and evaluation steps |
| Lab 04 | Explaining fine-tuning and LoRA, generating the initial set of 30 StillHour question-and-answer pairs, and identifying the bug where LoRA changed the base model in place |
| Lab 05 | Suggested code for the ingestion script, query rewriting with MMR and the Streamlit app; for the Knowledge Graph RAG activity, the code |
| Lab 06 | Debugging the agents, including suggested code fixes for `calc_agent.py` and `memory_agent.py` |
| Documentation | Helping structure lab READMEs and final reflection |

## Acknowledgements
The lab tasks and starter code are based on the PUSL3205 lab sheets , and the code was modified and extended for this portfolio.
The signed Generative AI Declaration is available in [GenAI-Declaration.pdf](GenAI-Declaration.pdf).