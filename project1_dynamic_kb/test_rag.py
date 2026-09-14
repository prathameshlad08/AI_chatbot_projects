from rag_chain import get_rag_response

question = "What is the Eiffel Tower made of and when was it built"
result = get_rag_response(question)

print("Question:", question)
print("\nAnswer:", result["answer"])
print("\nSources:", result["sources"])