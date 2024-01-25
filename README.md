# My_ESCO_RAG
Testing LLM RAG with ESCO ontology data

This is a simple RAG with GUI implementation to match ESCO skills and occupations to a given free text, e.g., CV or job advertisement. Simple GUI shows LLM-suggested skills/occupations and other potential skills/occupations from the vectorstore. Suggestions are recomputed as the user accepts/declines current skills/occupations.

![Screenshot 2024-01-25 162104](https://github.com/kauttoj/My_ESCO_RAG/assets/17804946/1453ce60-5ffe-439a-9c2c-301e13658611)

Repo also includes a simple GUI to test HEADAI model for ESCO and script to compute vectorstores from ESCO data.

Using the RAG requires openAI API token to use gpt-3.5 or gpt-4 models for picking relevant items. Retrieval can use both OpenAI or free Huggingface model. HEADAI model requires headai token. Both tokens are retrieved from your environmental variables (OPENAI_API_KEY and HEADAI_TOKEN).

Note: This is just a crude, unoptimized RAG without polishing and many parameters hard-coded.

-25.1.2024
