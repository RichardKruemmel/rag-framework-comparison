from typing import Any, Optional

from crewai import Agent, Task, Crew
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI

from frameworks.langchain import setup_langchain, setup_perf_langchain
from utils.utils import get_env_variable


class RetrieveTool(BaseTool):
    name: str = "retrieve_tool"
    description: str = "Nützlich für Fragen zu spezifischen Aspekten des Wahlprogramms."
    retrieval_qa: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.retrieval_qa = setup_langchain()

    def _run(self, aspect: str) -> str:
        return self.retrieval_qa.invoke(aspect)


class PerfRetrieveTool(BaseTool):
    name: str = "perf_retrieve_tool"
    description: str = "Nützlich für Fragen zu spezifischen Aspekten des Wahlprogramms."
    retrieval_qa: Optional[Any] = None

    def __init__(self):
        super().__init__()
        self.retrieval_qa = setup_perf_langchain()

    def _run(self, aspect: str) -> str:
        return self.retrieval_qa.invoke(aspect)


def setup_crew():
    tool_instance = RetrieveTool()
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
    )
    # Agent
    election_manifesto_expert = Agent(
        role="Experte für Wahlprogramme",
        goal="Verstehen und Interpretieren von Wahlprogrammen zur Beantwortung spezifischer Fragen",
        backstory=""""Du bist ein Experte für Wahlprogramme mit jahrelanger Erfahrung in der Analyse politischer Dokumente.
                Du hast die Fähigkeit, komplexe politische Texte zu durchdringen und klare, sachliche Antworten auf Fragen zu geben.
                Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz und faktisch korrekt.""",
        tools=[tool_instance],
        verbose=False,
        llm=llm,
        allow_delegation=False,
    )

    # Tasks
    task_qa = Task(
        description="""Beantworten die folgende Frage zum Wahlprogramm: {question}
                        Die Antwort sollte sachlich korrekt und auf dem Wahlprogramm basieren. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.""",
        expected_output="""Die endgültige Antwort sollte eine sachliche und korrekte Antwort auf die Frage sein, basierend auf dem Wahlprogramm.""",
        agent=election_manifesto_expert,
    )

    crew = Crew(agents=[election_manifesto_expert], tasks=[task_qa], verbose=False)

    return crew


def setup_perf_crew():
    tool_instance = PerfRetrieveTool()
    openai_api_key = get_env_variable("OPENAI_API_KEY")
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model="gpt-3.5-turbo",
        temperature=0.1,
    )
    # Agent
    election_manifesto_expert = Agent(
        role="Experte für Wahlprogramme",
        goal="Verstehen und Interpretieren von Wahlprogrammen zur Beantwortung spezifischer Fragen",
        backstory=""""Du bist ein Experte für Wahlprogramme mit jahrelanger Erfahrung in der Analyse politischer Dokumente.
                Du hast die Fähigkeit, komplexe politische Texte zu durchdringen und klare, sachliche Antworten auf Fragen zu geben.
                Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz und faktisch korrekt.""",
        tools=[tool_instance],
        verbose=False,
        llm=llm,
        allow_delegation=False,
    )

    # Tasks
    task_qa = Task(
        description="""Beantworten die folgende Frage zum Wahlprogramm: {question}
                        Die Antwort sollte sachlich korrekt und auf dem Wahlprogramm basieren. Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt. Verwende maximal drei Sätze und halte die Antwort kurz.""",
        expected_output="""Die endgültige Antwort sollte eine sachliche und korrekte Antwort auf die Frage sein, basierend auf dem Wahlprogramm.""",
        agent=election_manifesto_expert,
    )

    crew = Crew(agents=[election_manifesto_expert], tasks=[task_qa], verbose=False)

    return crew
