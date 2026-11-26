# Final Reflection – LLM Practical Portfolio

**Student:** HWS Sathsarani (10967109)  
**Module:** PUSL3205 – Large Language Models and Generative AI

## Overview
The six laboratory sessions followed a clear progression, from traditional machine learning to modern AI systems that can take actions.

| Lab | Topic | Main idea learned |
|---|---|---|
| 01 | Sentiment analysis with traditional ML | Text must be converted into numbers, and simple models cannot understand context |
| 02 | LLM as a product-development partner | Well-structured prompts turn an LLM into a useful design assistant |
| 03 | Tiny Transformer from scratch | How a language model learns to predict the next token |
| 04 | Fine-tuning with LoRA | How to adapt a pre-trained model efficiently, and the limits of doing so |
| 05 | Retrieval-Augmented Generation | How to ground answers in real documents |
| 06 | AI agents | How a model can use tools, memory and external services to complete tasks |

Each lab built on the previous one. Together they showed how a modern AI assistant is put together, from the model itself to the knowledge it uses and the actions it can take.

## Most Engaging Labs
Labs 05 and 06 were the most interesting for me, because they were completely new and moved from models that only generate text to systems that do useful work. In Lab 05, the University Knowledge Assistant answered questions from the NSBM Computing Award Handbook and showed the page each answer came from. In Lab 06, the agents opened applications, remembered information between sessions and created a real event in Google Calendar. This agentic approach, where the model decides which tool to use, is the area I most want to explore further.

## Most Challenging Lab
Lab 04 was the most difficult, because the mechanism of fine-tuning was hard to understand in depth. It involved several new ideas at once: LoRA adapters, the chat template, BF16 precision, gradient accumulation and the training loss. Two practical problems made it harder. First, applying LoRA changed the base model in place, so the "before" and "after" answers were identical until the original answers were recorded before training. Second, running the same notebook twice with the same seed gave different results, because GPU calculations are not fully deterministic. Working through these problems helped me understand what fine-tuning actually changes inside a model.

## Understanding How LLMs Work
Large language models are deep learning systems built mainly on the Transformer architecture. They predict the next token in a sequence based on statistical patterns learned from large amounts of text and on the context of the input. Building a Tiny Transformer in Lab 03 made this concrete. With only 834,401 parameters, the model learned spelling, punctuation and typical story openings, but it lost meaning after one or two sentences.

Comparing it with Lab 01 also showed why the Transformer was such an important step. The Bag-of-Words model ignores word order, so it could not understand phrases such as "not bad". Self-attention allows each token to use the context of the tokens before it. This explained to me why larger models trained on far more data can produce fluent and useful text, and why they still do not truly "understand" it.

## Fine-Tuning or RAG
After Labs 04 and 05, I prefer RAG for applications that depend on factual knowledge. Fine-tuning on 30 examples changed the model's style, but it answered only one of five reworded questions correctly and invented facts such as a 10% discount instead of 15%. RAG answered from the actual documents, showed its sources and refused questions that the documents did not cover. RAG also allows new documents to be added without retraining. Fine-tuning remains useful for teaching a model a particular style or format, and the two methods can be combined.

## Hallucination
An AI hallucination happens when a model confidently generates false, fabricated or nonsensical information and presents it as fact. The labs showed this clearly. In Lab 04, both models claimed that the StillHour Hub could play Spotify music, and the fine-tuned model said the Hub stores audio in the cloud, which was the opposite of the training data. The most surprising part was that fine-tuning made the hallucinations sound more convincing, because they used the same confident tone as the correct answers. In Lab 05, grounding the model in retrieved documents and giving it a fixed "not found" reply reduced this problem. I also learned that automatic metrics such as BLEU and chrF cannot detect hallucination, because they only measure word overlap, so human checking is still needed.

## Skills Gained
Every lab introduced new practical skills:
- **Machine learning and NLP:** text preprocessing, Bag-of-Words, Logistic Regression, SVM and evaluation metrics (Lab 01)
- **Prompt engineering:** structuring prompts with a role, context, task, format and constraints (Lab 02)
- **Deep learning with PyTorch:** building and training a Transformer, and measuring loss and perplexity (Lab 03)
- **Hugging Face, PEFT and TRL:** fine-tuning with LoRA on a local GPU and evaluating with BLEU and chrF (Lab 04)
- **LangChain, Ollama, ChromaDB and Streamlit:** building a complete local RAG application (Lab 05)
- **AI agents:** tool calling, SQLite memory and the Google Calendar API with OAuth 2.0 (Lab 06)
- **Software practice:** virtual environments, debugging, Git and GitHub, and keeping credentials out of a public repository

## Use of Generative AI Tools
Generative AI was used as an assistant throughout the labs: to plan and organise the work, to explain difficult concepts, to help debug errors and to refine the documentation. I modified the lab sheet code and organised it into my own structure. All experiments, results and screenshots come from my own runs.

An important lesson was that AI suggestions must always be tested. Some suggested fixes did not work the first time. For example, the memory search in Lab 06 still failed until the real cause, a missing stop-word list, was found by checking the program's output carefully. Using AI tools well means checking their answers, understanding the code before using it and declaring their use honestly.

## Future Application
I want to build assistants and agents like the ones in Labs 05 and 06. A natural next step would be a university assistant that combines RAG for handbook questions, memory for each student's details and tools such as a calendar for deadlines. This connects directly to my goal of becoming a machine learning engineer. The labs also showed what a real product would need: reliable evaluation, protection against hallucination, careful handling of personal data and secure access to external services.

## Conclusion
This module changed how I see LLMs. I now understand them as next-token predictors built on the Transformer, which become much more useful when they are grounded in real documents and connected to tools. The most valuable lessons came from the problems rather than the successful runs: hallucinations, unreliable tool calls and non-repeatable results. These showed that building a trustworthy AI system needs careful testing, evaluation and responsible design, not just a powerful model.