import os
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

load_dotenv()
st.set_page_config(page_title="AI Blog Content Creator", page_icon="✍️", layout="wide")


def build_crew(topic: str) -> Crew:
    llm = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    planner = Agent(
        llm=llm,
        role="Content Planner",
        goal="Plan engaging and factually accurate content on {topic}",
        backstory=(
            "You're working on planning a blog article about the topic: {topic}. "
            "You collect information that helps the audience learn something and "
            "make informed decisions. Your work is the basis for the Content Writer "
            "to write an article on this topic."
        ),
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        llm=llm,
        role="Content Writer",
        goal="Write insightful and factually accurate opinion piece about the topic: {topic}",
        backstory=(
            "You're writing a new opinion piece about the topic: {topic}. You base "
            "your writing on the work of the Content Planner, who provides an outline "
            "and relevant context about the topic."
        ),
        allow_delegation=False,
        verbose=True,
    )

    editor = Agent(
        llm=llm,
        role="Editor",
        goal="Edit a given blog post to align with the writing style of the organization.",
        backstory=(
            "You are an editor who receives a blog post from the Content Writer. "
            "Your goal is to review the blog post to ensure that it follows journalistic "
            "best practices, provides balanced viewpoints when providing opinions or "
            "assertions, and avoids major controversial topics or opinions when possible."
        ),
        allow_delegation=False,
        verbose=True,
    )

    plan = Task(
        description=(
            "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
            "2. Identify the target audience, considering their interests and pain points.\n"
            "3. Develop a detailed content outline including an introduction, key points, and a call to action.\n"
            "4. Include SEO keywords and relevant data or sources."
        ),
        expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and resources.",
        agent=planner,
    )

    write = Task(
        description=(
            "1. Use the content plan to craft a compelling blog post on {topic}.\n"
            "2. Incorporate SEO keywords naturally.\n"
            "3. Sections and subtitles should be properly named in an engaging manner.\n"
            "4. Ensure the post is structured with an engaging introduction, insightful body, and a summarizing conclusion.\n"
            "5. Proofread for grammatical errors and alignment with the brand's voice."
        ),
        expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
        agent=writer,
    )

    edit = Task(
        description="Proofread the given blog post for grammatical errors and alignment with the brand's voice.",
        expected_output="A well-written blog post in markdown format, ready for publication, each section should have 2 or 3 paragraphs.",
        agent=editor,
    )

    return Crew(
        agents=[planner, writer, editor],
        tasks=[plan, write, edit],
        process=Process.sequential,
        verbose=True,
    )


def generate_blog_content(topic: str) -> str:
    api_key = st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY. Enter it in the sidebar or add it to your environment/.env file before running the app.")

    os.environ["OPENAI_API_KEY"] = api_key
    crew = build_crew(topic)
    result = crew.kickoff(inputs={"topic": topic})
    return str(result)


st.title("AI Blog Content Generator")
st.write("Enter a topic and let the agents plan, write, and edit a polished blog post for you.")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Saved Content")
    if st.session_state.history:
        for item in reversed(st.session_state.history):
            with st.expander(item["topic"], expanded=False):
                st.markdown(item["content"])
    else:
        st.caption("No generated content yet.")

topic = st.text_input(
    "Topic",
    placeholder="e.g. AI in healthcare",
    help="Enter the topic you want the agents to write about.",
)

col1, col2 = st.columns([1, 1])
with col1:
    generate_button = st.button("Generate Content", use_container_width=True)
with col2:
    clear_button = st.button("Clear", use_container_width=True)

if clear_button:
    st.session_state.pop("generated_content", None)

if generate_button:
    if not topic.strip():
        st.warning("Please enter a topic before generating content.")
    else:
        with st.spinner("The agents are working on your content..."):
            try:
                result = generate_blog_content(topic.strip())
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
            else:
                st.session_state.generated_content = result
                st.session_state.history.append({"topic": topic.strip(), "content": result})
                st.success("Content generated successfully!")

if "generated_content" in st.session_state and st.session_state.generated_content:
    st.markdown("## Generated Output")
    st.markdown(st.session_state.generated_content)
