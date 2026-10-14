"""
LangGraph routing: a smart query router (LangGraph + Groq).

Classifies an incoming question (physics / math / nature / general) and then
routes it to a matching "expert" prompt via a LangGraph conditional edge, so each
query is answered by a specialist persona. Wrapped in a Gradio web UI.

The classifier uses structured output: the model fills in a Pydantic
ClassificationResult (one of the four categories plus a confidence score) rather
than free text that has to be parsed. A classification below the confidence
threshold is not trusted: the query goes to the general assistant instead of a
specialist, as a safe fallback.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python query_router.py

It answers one sample query, then launches a Gradio app at
http://127.0.0.1:7860 (press Ctrl+C to stop).
"""

import sys
from pathlib import Path
from typing import Literal, Optional, TypedDict, Dict

import gradio as gr
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

# Below this confidence the classifier's choice is not trusted and the query is
# answered by the general assistant instead of a specialist.
CONFIDENCE_THRESHOLD = 0.5

class QueryInput(BaseModel):
    query: str = Field(description="The user's original query")


class ClassificationResult(BaseModel):
    category: Literal["physics", "math", "nature", "general"] = Field(
        description="The classified category of the query"
    )
    confidence: float = Field(
        description="Confidence score for the classification", 
        ge=0.0, 
        le=1.0
    )


class Response(BaseModel):
    answer: str = Field(description="The answer to the user's query")
    category_used: Literal["physics", "math", "nature", "general"] = Field(
        description="The category used to generate the answer"
    )


class State(TypedDict):
    input: QueryInput
    classification: Optional[ClassificationResult]
    response: Optional[Response]


class QueryRouter:
    def __init__(self, model_name="openai/gpt-oss-120b", confidence_threshold=CONFIDENCE_THRESHOLD):
        # gpt-oss reasons before it answers and those tokens count towards max_tokens.
        # Groq's default cap can cut long answers off. The expert answers here are
        # long but the prompts are short, so 6000 fits while keeping prompt +
        # max_tokens within the free tier's 8,000 tokens-per-minute budget.
        self.llm = ChatGroq(model_name=model_name, max_tokens=6000, max_retries=10)
        self.confidence_threshold = confidence_threshold

        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classifier. Classify the query into exactly one category; do NOT answer it.
            physics = forces, motion, energy, matter, space and the laws that describe them
            math = calculations, equations, proofs and other mathematical concepts
            nature = living things, ecosystems, weather, geology and other natural phenomena
            general = anything else
            Also rate your confidence in the classification from 0 (a guess) to 1 (certain)."""),
            ("human", "{query}")
        ])
        # Structured output: the model fills in a ClassificationResult instead of
        # replying in free text, so the category is always one of the four valid values.
        self.classifier = self.classification_prompt | self.llm.with_structured_output(ClassificationResult)

        self.physics_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a physics expert. Provide accurate, detailed answers to physics questions.
            Focus on explaining physical laws, theories, and phenomena in a clear and educational manner.
            Use scientific terminology appropriately and include relevant formulas when needed."""),
            ("human", "{query}")
        ])
        
        self.math_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a mathematics expert. Provide accurate, detailed answers to math questions.
            Focus on explaining mathematical concepts, proofs, and problem-solving techniques.
            Show step-by-step solutions when appropriate and use precise mathematical notation."""),
            ("human", "{query}")
        ])
        
        self.nature_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a natural sciences expert. Provide accurate, detailed answers about the natural world.
            Focus on explaining biological systems, ecological relationships, geological processes, and other natural phenomena.
            Use scientific terminology appropriately and emphasize the interconnectedness of natural systems."""),
            ("human", "{query}")
        ])
        
        self.general_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable assistant. Provide helpful, accurate information on general topics.
            Aim to be informative and educational while keeping explanations accessible to a general audience.
            Draw from a broad knowledge base to give well-rounded answers."""),
            ("human", "{query}")
        ])
    
        self.setup_graph()

    def classify_query(self, state: State) -> State:
        query = state["input"].query
        try:
            classification = self.classifier.invoke({"query": query})
        except ValueError:
            # Pydantic validation and output-parsing errors (e.g. a confidence
            # outside [0, 1]) are ValueErrors: treat the query as unclassified.
            classification = None
        if classification is None:
            # No usable classification: zero confidence sends the query to the general assistant.
            classification = ClassificationResult(category="general", confidence=0.0)

        return {
            **state,
            "classification": classification
        }
    
    def route_to_physics(self, state: State) -> State:
        query = state["input"].query
        chain = self.physics_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="physics"
            )
        }
    
    def route_to_math(self, state: State) -> State:
        query = state["input"].query
        chain = self.math_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="math"
            )
        }
    
    def route_to_nature(self, state: State) -> State:
        query = state["input"].query
        chain = self.nature_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="nature"
            )
        }
    
    def route_to_general(self, state: State) -> State:
        query = state["input"].query
        chain = self.general_prompt | self.llm
        result = chain.invoke({"query": query})
        
        return {
            **state,
            "response": Response(
                answer=result.content,
                category_used="general"
            )
        }
    
    def decide_route(self, state: State) -> Literal["physics", "math", "nature", "general"]:
        classification = state["classification"]
        # A low-confidence classification falls back to the general assistant.
        if classification.confidence < self.confidence_threshold:
            return "general"
        return classification.category
    
    def setup_graph(self):
        self.graph = StateGraph(State)
        
        self.graph.add_node("classification_node", self.classify_query)
        self.graph.add_node("physics_node", self.route_to_physics)
        self.graph.add_node("math_node", self.route_to_math)
        self.graph.add_node("nature_node", self.route_to_nature)
        self.graph.add_node("general_node", self.route_to_general)
        
        self.graph.add_edge(START, "classification_node")
        self.graph.add_conditional_edges(
            "classification_node",
            self.decide_route,
            {
                "physics": "physics_node",
                "math": "math_node",
                "nature": "nature_node",
                "general": "general_node"
            }
        )
        
        self.graph.add_edge("physics_node", END)
        self.graph.add_edge("math_node", END)
        self.graph.add_edge("nature_node", END)
        self.graph.add_edge("general_node", END)
        
        self.graph_instance = self.graph.compile()
    
    def process_query(self, query: str) -> Dict:
        initial_state = {"input": QueryInput(query=query), "classification": None, "response": None}
        result = self.graph_instance.invoke(initial_state)
        return {
            "query": query,
            # The expert that actually answered; differs from predicted_category
            # when a low-confidence classification fell back to "general".
            "category": result["response"].category_used,
            "predicted_category": result["classification"].category,
            "confidence": result["classification"].confidence,
            "answer": result["response"].answer
        }

    def describe_route(self, result: Dict) -> str:
        """One-line, human-readable summary of how a query was routed."""
        summary = f"Query classified as: {result['predicted_category']} (confidence: {result['confidence']:.2f})"
        if result["category"] != result["predicted_category"]:
            summary += (f", below the {self.confidence_threshold:.2f} threshold, "
                        f"so the {result['category']} assistant answered")
        return summary


if __name__ == "__main__":
    router = QueryRouter(model_name="openai/gpt-oss-120b")
    response = router.process_query("What is the theory of relativity?")
    print(f"Query: {response['query']}")
    print(router.describe_route(response))
    print(f"\nAnswer:\n{response['answer']}\n")


    def process_query(query):
        yield "", "", "", "", "Processing query..."

        result = router.process_query(query)

        physics_output = result["answer"] if result["category"] == "physics" else ""
        math_output = result["answer"] if result["category"] == "math" else ""
        nature_output = result["answer"] if result["category"] == "nature" else ""
        general_output = result["answer"] if result["category"] == "general" else ""

        status = router.describe_route(result)

        yield physics_output, math_output, nature_output, general_output, status

    with gr.Blocks(title="Query Router Interface") as demo:
        gr.Markdown("# Smart Query Router")
        gr.Markdown("Enter your question and the system will route it to the appropriate expert.")
        
        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Enter your query",
                    placeholder="e.g., How does gravity work?",
                    lines=5
                )
                submit_btn = gr.Button("Submit", variant="primary")
                status = gr.Markdown("")
            
            with gr.Column(scale=2):
                with gr.Tab("Physics"):
                    physics_output = gr.Markdown(label="Physics Expert")
                with gr.Tab("Mathematics"):
                    math_output = gr.Markdown(label="Mathematics Expert")
                with gr.Tab("Nature"):
                    nature_output = gr.Markdown(label="Nature Expert")
                with gr.Tab("General"):
                    general_output = gr.Markdown(label="General Knowledge")
        
        submit_btn.click(
            fn=process_query,
            inputs=query_input,
            outputs=[physics_output, math_output, nature_output, general_output, status],
            api_name="query"
        )
        
        examples = [
            ["What is the theory of relativity?"],
            ["How do I solve quadratic equations?"],
            ["Why do leaves change color in autumn?"],
            ["Who was the first president of the United States?"]
        ]
        gr.Examples(examples=examples, inputs=query_input)

    demo.launch()
