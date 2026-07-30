from dotenv import load_dotenv
from crewai import Task, Agent, Crew, Process
import os

load_dotenv()


# API KEY
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
# MODEL NAME 
llm ="gpt-5.4-mini" 






# ===========================================================================
#                                AGENTS
# ===========================================================================

researcher = Agent(
    role="senior Research Analyst",
    goal="Gather accurate and relevant information for content",
    backstory="""
    You are an expert researcher skilled at finding concise,
    factual, and useful insights for professional writing.
    """,
    verbose=True,
    allow_delegation=False,
    llm= llm,
    tracing=True,
)

writer = Agent(
    role="Senior Content Writer",
    goal="Create polished, engaging, and professional content.",
    backstory="""
    You are profesional writer known for crafting 
    high-quality blog posts and articles.
    """,
    verbose=True,
    allow_delegation=False,
    llm= llm,
    tracing=True,
)

critic = Agent(
    role="Edidorial Critic",
    goal="Review and improve content quality.",
    backstory="""
    You are a strict editor who reviews content
    for clarity, tone, structure, and readability.
    """,
    verbose=True,
    allow_delegation=False,
    llm= llm,
    tracing=True,
)

# ===========================================================================
# USER TASK
# ===========================================================================

# Ask for topic before creating task descriptions that reference it
# so the f-string has a defined value at runtime.
topic = input("Enter a topic for the blog post: ")

research_task = Task(
    description=f"""
    Research the following topic:
    {topic}

    Focus on:
    - Key benefits
    - Real-world use cases
    - Industry impact
    - Challenges
    - Future opportunities

    Keep findings structured and concise.
    """,
    expected_output="""
    Structured research notes with key insights and examples.
    """,
    agent = researcher,
)

writing_task = Task(
    description="""
    using the provided research,
    write a polished professional blog posts.


    Requirements:
    - Strong introduction
    - Clear structure
    - Engaging tone
    - Practical examples
    - Strong conclusion
    """,
    expected_output="""
    A complete blog post ready for publishing
    """,
    
    agent = writer,
)

review_task = Task(
    description=f"""
    Critically review the blog post:
   

    Evalute:
    - Clarity
    - Accuracy
    - Readability
    - Structure
    - Professional tone
    - Completeness

    If improvements are needed,
    provide specific actionable feedback.
    """,
    expected_output="""
    Editorial review with strengths, weakness,
    and improvement suggestions.
    """,
    agent = critic,
)

# ===========================================================================
# CREW
# ===========================================================================


crew = Crew(
    agents =[
        researcher,
        writer,
        critic,
    ],
    tasks=[
        research_task,
        writing_task,
        review_task
    ],
    process=Process.sequential,
    verbose=True,
)

# ===========================================================================
# EXECUTE
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("\n" + "=" * 80)

    result = crew.kickoff(inputs={"topic": topic})
    print(result)

