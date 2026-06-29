"""
LangGraph intro: a sequential text-analysis pipeline (LangGraph + Groq).

Builds a simple three-node LangGraph that processes text in order:
classify it (News / Blog / Research / Other) -> extract entities -> summarize.
The same TextProcessor is also wrapped in a small Gradio web UI.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt_05_langgraph_intro.py

It prints the analysis for a sample text, then launches a Gradio app at
http://127.0.0.1:7860 (press Ctrl+C to stop).
"""

import sys
from pathlib import Path
from typing import TypedDict, List

import gradio as gr
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

class TextProcessor:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_retries=10)
        self.graph = self.create_graph()

    def create_graph(self):
        graph = StateGraph(State)
        graph.add_node("classification_node", self.classification_node)
        graph.add_node("entity_extraction", self.entity_extraction_node)
        graph.add_node("summarization", self.summarization_node)
        graph.set_entry_point("classification_node")
        graph.add_edge("classification_node", "entity_extraction")
        graph.add_edge("entity_extraction", "summarization")
        graph.add_edge("summarization", END)
        return graph.compile()

    def classification_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Classify the following text into one of the categories: News, Blog, Research, or Other. Do not write anything else except of these words.\n\nText:{text}\n\nCategory:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        classification = self.llm.invoke([message]).content.strip()
        return {"classification": classification}

    def entity_extraction_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        entities = self.llm.invoke([message]).content.strip().split(", ")
        return {"entities": entities}

    def summarization_node(self, state):
        prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
        )
        message = HumanMessage(content=prompt.format(text=state["text"]))
        summary = self.llm.invoke([message]).content.strip()
        return {"summary": summary}

    def process(self, text):
        state_input = {"text": text}
        result = self.graph.invoke(state_input)
        print("Classification:", result["classification"])
        print("\nEntities:", result["entities"])
        print("\nSummary:", result["summary"])
        return result


class App:
    def __init__(self):
        self.processor = TextProcessor()
        self.interface = self.setup_interface()

    def setup_interface(self):
        with gr.Blocks(title="Text Analysis Tool") as interface:
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter text to analyze...",
                        lines=10
                    )
                    analyze_btn = gr.Button("Analyze Text")

                    sample_btn = gr.Button("Load Sample Text")

                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("Classification"):
                            classification_output = gr.Textbox(label="Classification Result", interactive=False)

                        with gr.TabItem("Entities"):
                            entities_output = gr.Dataframe(
                                headers=["Entity"],
                                label="Extracted Entities",
                                interactive=False
                            )

                        with gr.TabItem("Summary"):
                            summary_output = gr.Textbox(label="Summary", interactive=False)

            analyze_btn.click(
                fn=self.analyze_text,
                inputs=text_input,
                outputs=[classification_output, entities_output, summary_output]
            )

            sample_btn.click(
                fn=self.load_sample_text,
                inputs=None,
                outputs=text_input
            )

        return interface

    def analyze_text(self, text):
        if not text or text.strip() == "":
            return "No text provided", [["No entities found"]], "No text to summarize"

        result = self.processor.process(text)

        entities_list = [[entity] for entity in result["entities"]]
        if not entities_list:
            entities_list = [["No entities found"]]

        return result["classification"], entities_list, result["summary"]

    def load_sample_text(self):
        sample_text = """
OpenAI has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""
        return sample_text

    def launch(self, share=False):
        self.interface.launch(share=share)


if __name__ == "__main__":
    processor = TextProcessor()
    sample_text = """
    The World Health Organization (WHO) has published a new report on climate change and its impact on global health systems. According to the findings, rising temperatures are leading to increased spread of infectious diseases, particularly in tropical regions.

    The report also highlights that extreme weather events such as floods, hurricanes, and droughts are causing significant disruptions to healthcare infrastructure in vulnerable communities. Researchers from Oxford University and the CDC collaborated on the study, which analyzed data from 45 countries over a period of 15 years.

    WHO Director-General emphasized that healthcare systems need to become more resilient and adaptable to climate-related challenges, recommending that countries allocate at least 5% of their health budgets to climate adaptation measures by 2030.
    """
    processor.process(sample_text)
    

    app = App()
    app.launch()
