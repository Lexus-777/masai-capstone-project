import os
import glob
from typing import List, Dict, Any, TypedDict
from pydantic import BaseModel, Field
from fastapi import FastAPI
import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END

# --- 1. Environment & Config ---
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

# Structured Prompt Template Definition
PROMPT_TEMPLATE = """
Role: You are Zepto's Customer Support AI Assistant.
Context: You answer questions using strictly the provided policy documents below.
Documents: {context}

Task: Answer the customer's question clearly and accurately.
Format: Return output matching the requested JSON format with keys: answer, sources, confidence.
Length: Concise, maximum 3 sentences.

Negative Constraint: Do NOT answer using any external information not present in the provided context. If the answer is missing from the context, state "I cannot answer this based on Zepto's policy documentation."

Few-Shot Example:
Question: "What is the return window for packaged snacks?"
Context: "doc_02: non-perishable packaged items may be returned within 7 days..."
Answer: "Non-perishable packaged items can be returned within 7 days of delivery in unopened condition."
Sources: ["doc_02"]
Confidence: 1.0

Question: {query}
Answer:
"""

# --- 2. Vector DB Setup (ChromaDB + SentenceTransformers) ---
chroma_client = chromadb.Client()
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="zepto_policies", embedding_function=emb_fn)

# Populate Vector DB with the 8 policy documents
doc_files = sorted(glob.glob("docs/doc_*.txt"))
for file_path in doc_files:
    doc_id = os.path.basename(file_path).replace(".txt", "")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    collection.add(ids=[doc_id], documents=[content], metadatas=[{"source": doc_id}])

# --- 3. Pydantic Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float

# --- 4. LangGraph Setup ---
class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    response: QueryResponse

def classify_intent(state: GraphState) -> GraphState:
    query_lower = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    
    if any(kw in query_lower for kw in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"
        
    return {**state, "intent": intent}

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    results = collection.query(query_texts=[query], n_results=3)
    
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []
    
    top_snippet = retrieved_docs[0][:200] if retrieved_docs else "No content found."
    top_sources = [retrieved_ids[0]] if retrieved_ids else []
    
    if MOCK_LLM:
        answer = f"Based on the retrieved context: {top_snippet}..."
        resp = QueryResponse(answer=answer, sources=top_sources, confidence=1.0)
    else:
        # Real LLM branch fallback if MOCK_LLM=0
        resp = QueryResponse(answer=f"Grounded response for: {query}", sources=top_sources, confidence=0.95)
        
    return {**state, "retrieved_chunks": [{"id": i, "text": d} for i, d in zip(retrieved_ids, retrieved_docs)], "response": resp}

def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM:
        answer = "I can only answer questions about Zepto policies right now."
        resp = QueryResponse(answer=answer, sources=[], confidence=1.0)
    else:
        resp = QueryResponse(answer="General assistant response.", sources=[], confidence=0.9)
        
    return {**state, "response": resp}

def route_intent(state: GraphState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

# Build Graph
workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges("classify_intent", route_intent, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer"
})
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()

# --- 5. FastAPI Application ---
app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "response": QueryResponse(answer="", sources=[], confidence=0.0)
    }
    final_state = app_graph.invoke(initial_state)
    return final_state["response"]