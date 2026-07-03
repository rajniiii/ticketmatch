# TicketMatch

TicketMatch is a support ticket classification project that uses LangChain and Hugging Face embeddings. The workflow loads CSV training and test files, builds vector stores with both OpenAI and local Gemma embeddings, and evaluates classification performance using semantic search and prompt-based methods.

## Workflow

1. Load ticket data from `ticket_train.csv` and `ticket_test.csv`.
2. Read OpenAI API key from `api_key.txt` (excluded from git) and set `OPENAI_API_KEY`.
3. Create embeddings:
   - `OpenAIEmbeddings` for high-quality cloud embeddings.
   - `HuggingFaceEmbeddings` for local Gemma embeddings using a local `embeddinggemma-300m` model.
4. Build Chroma vector stores in `chroma_openai/` and `chroma_gemma/` from training ticket text.
5. Run three classification workflows:
   - Zero-shot classification with an LLM prompt.
   - KNN classification using vector similarity search over the Chroma store.
   - Few-shot prompt classification with retrieved examples from the vector store.
6. Print detailed classification reports comparing accuracy, precision, recall, and F1.

## Notes

- This project uses `langchain_openai`, `langchain_huggingface`, and `langchain_chroma`.
- `api_key.txt` is intentionally ignored because it holds sensitive API keys.
- `chroma_openai`, `chroma_gemma`, and local model directories are also ignored because they store runtime-generated caches and large model files.
- The project is designed to work with CSV ticket data and embed ticket text for semantic classification.

## Run

```bash
python rshenkar_hw8.py
```

## Files

- `rshenkar_hw8.py` — main classification script.
- `rshenkar_requirements.txt` — required Python packages.
- `ticket_train.csv` — training dataset.
- `ticket_test.csv` — test dataset.
- `.gitignore` — ignores secret and runtime cache files.
