"""
Self-Discover agent: compose a reasoning structure, then solve (LangGraph + Groq).

Implements the Self-Discover technique as a four-node LangGraph pipeline:
SELECT relevant reasoning modules -> ADAPT them to the task -> STRUCTURE them into
a step-by-step JSON plan -> REASON over that plan to produce the answer.

The four prompts are defined locally (see the *_PROMPT constants) so the script is
self-contained and needs no network prompt-hub access.

How to run
----------
1. Create a `.env` file next to this script with your Groq API key:
       GROQ_API_KEY=your_key_here
2. Install dependencies:  pip install -r requirements.txt
3. Run:                   python prompt_07_self_discover.py

It prints a direct model answer to a sample task, then streams the Self-Discover
pipeline's state at each step. This makes several model calls, so on Groq's free
tier it may pause to respect rate limits.
"""

import sys
from pathlib import Path
from typing import List, Optional

from typing_extensions import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

# Force UTF-8 stdout so model output with special characters does not crash on Windows.
sys.stdout.reconfigure(encoding="utf-8")

# Load GROQ_API_KEY from a .env file located next to this script.
load_dotenv(Path(__file__).resolve().parent / ".env")


# The four Self-Discover prompts, defined locally so the script is self-contained
# (these were previously fetched from LangChain Hub via hub.pull, which is now
# disabled by default). Content matches hwchase17/self-discovery-{select,adapt,
# structure,reasoning}.
SELECT_PROMPT = 'Select several reasoning modules that are crucial to utilize in order to solve the given task:\n\nAll reasoning module descriptions:\n{reasoning_modules}\n\nTask: {task_description}\n\nSelect several modules are crucial for solving the task above:\n'

ADAPT_PROMPT = 'Rephrase and specify each reasoning module so that it better helps solving the task:\n\nSELECTED module descriptions:\n{selected_modules}\n\nTask: {task_description}\n\nAdapt each reasoning module description to better solve the task:\n'

STRUCTURE_PROMPT = 'Operationalize the reasoning modules into a step-by-step reasoning plan in JSON format:\n\nHere\'s an example:\n\nExample task:\n\nIf you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.\n\nExample reasoning structure:\n\n{{\n    "Position after instruction 1":\n    "Position after instruction 2":\n    "Position after instruction n":\n    "Is final position the same as starting position":\n}}\n\nAdapted module description:\n{adapted_modules}\n\nTask: {task_description}\n\nImplement a reasoning structure for solvers to follow step-by-step and arrive at correct answer.\n\nNote: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can fill it out and arrive at the correct conclusion for tasks like this'

REASONING_PROMPT = 'Follow the step-by-step reasoning plan in JSON to correctly solve the task. Fill in the values following the keys by reasoning specifically about the task given. Do not simply rephrase the keys.\n    \nReasoning Structure:\n{reasoning_structure}\n\nTask: {task_description}'


class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    answer: Optional[str]


class SelfDiscoveryAgent:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model = ChatGroq(model_name=model_name, max_retries=10)
        
        self.select_chain = self.setup_select_chain()
        self.adapt_chain = self.setup_adapt_chain()
        self.structure_chain = self.setup_structure_chain()
        self.reasoning_chain = self.setup_reasoning_chain()
        
        self.graph = self.build_graph()
    
    def setup_select_chain(self):
        select_prompt = PromptTemplate.from_template(SELECT_PROMPT)
        return select_prompt | self.model | StrOutputParser()

    def setup_adapt_chain(self):
        adapt_prompt = PromptTemplate.from_template(ADAPT_PROMPT)
        return adapt_prompt | self.model | StrOutputParser()

    def setup_structure_chain(self):
        structure_prompt = PromptTemplate.from_template(STRUCTURE_PROMPT)
        return structure_prompt | self.model | StrOutputParser()

    def setup_reasoning_chain(self):
        reasoning_prompt = PromptTemplate.from_template(REASONING_PROMPT)
        return reasoning_prompt | self.model | StrOutputParser()
    
    def select(self, state: dict) -> dict:
        return {"selected_modules": self.select_chain.invoke(state)}
    
    def adapt(self, state: dict) -> dict:
        return {"adapted_modules": self.adapt_chain.invoke(state)}
    
    def structure(self, state: dict) -> dict:
        return {"reasoning_structure": self.structure_chain.invoke(state)}
    
    def reason(self, state: dict) -> dict:
        return {"answer": self.reasoning_chain.invoke(state)}
    
    def build_graph(self) -> StateGraph:
        graph = StateGraph(SelfDiscoverState)
        
        graph.add_node("select", self.select)
        graph.add_node("adapt", self.adapt)
        graph.add_node("structure", self.structure)
        graph.add_node("reason", self.reason)
        
        graph.add_edge(START, "select")
        graph.add_edge("select", "adapt")
        graph.add_edge("adapt", "structure")
        graph.add_edge("structure", "reason")
        graph.add_edge("reason", END)
        
        return graph.compile()
    
    def solve(self, task_description: str, reasoning_modules: List[str]) -> None:
        reasoning_modules_str = "\n".join(reasoning_modules)
        initial_state = {
            "task_description": task_description,
            "reasoning_modules": reasoning_modules_str
        }
        config = {"configurable": {"thread_id": "1"}}

        for state in self.graph.stream(initial_state, config):
            print(state)
            print("\n")


if __name__ == "__main__":
    reasoning_modules = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "6. What are the potential risks and drawbacks of each solution?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        "34. Given the problem specification and the current best solution, have a guess about other possible solutions.",
        "35. Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?",
        "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?",
        "37. Ignoring the current best solution, create an entirely new solution to the problem.",
        "39. Let's make a step by step plan and implement it with good notation and explanation.",
    ]
    task_example_1 = "Lisa has 10 apples. She gives 3 apples to her friend and then buys 5 more apples from the store. How many apples does Lisa have now?"
    task_example_2 = """This SVG path element <path d="M 55.57,80.69 L 57.38,65.80 M 57.38,65.80 L 48.90,57.46 M 48.90,57.46 L
    45.58,47.78 M 45.58,47.78 L 53.25,36.07 L 66.29,48.90 L 78.69,61.09 L 55.57,80.69"/> draws a:
    (A) circle (B) heptagon (C) hexagon (D) kite (E) line (F) octagon (G) pentagon(H) rectangle (I) sector (J) triangle"""
    task_example_3 = """ 
    A school organizes a fundraiser selling tickets for $5 each. On Monday, they sold 45 tickets. On Tuesday, they sold twice as many tickets as Monday. On Wednesday, they sold 15 fewer tickets than Tuesday. Then, a local business donated an amount equal to half of the total sales so far. If the school needs $1,200 to reach their fundraising goal, how much more money do they still need to raise?
    """
    task_example_4 = """A new crypto coin has the following properties:

On day 1, there are 1,000 coins in circulation.
Each month, the number of coins in circulation increases by 20%.
At the start of month 10, the developers burn (remove from circulation) 30% of all coins.
Each month after that, the number of coins continues to increase by 20%.

How many coins will be in circulation at the end of month 15?"""

    agent = SelfDiscoveryAgent()
    print(agent.model.invoke(task_example_4))
    agent.solve(task_example_4, reasoning_modules)
