import subprocess
import click
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from textwrap import dedent


def get_git_diff():
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return "Error: Unable to get git diff. Make sure you're in a git repository and have staged changes."


def generate_commit_message(diff):
    llm = ChatOllama(model="starcoder2:15b")

    prompt = PromptTemplate(
        input_variables=["diff"],
        template=dedent(
            """\
        Based on the following git diff, generate a commit message in the style of conventional commits.
        The message should describe all of the changes and their purpose. 
        The message should have a type, an optional scope, and a description.
        Types include: feat, fix, docs, style, refactor, perf, test, chore.
        The output should only contain the output and nothing else.

        Git Diff:
        {diff}

        Commit Message:
        """
        ),
    )
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"diff": diff})
    return response.strip()


@click.command()
@click.option(
    "--preview", is_flag=True, help="Preview the commit message without committing"
)
def main(preview):
    """Generate a commit message based on staged changes using Ollama and LangChain."""
    diff = get_git_diff()

    if not diff or diff.startswith("Error"):
        click.echo(diff or "No changes detected.")
        return

    commit_message = generate_commit_message(diff)

    click.echo("Generated commit message:")
    click.echo(commit_message)

    if preview:
        click.echo("Preview mode: No changes committed.")
    else:
        if click.confirm("Do you want to commit with this message?", default=True):
            try:
                subprocess.run(["git", "commit", "-m", commit_message], check=True)
                click.echo("Changes committed successfully.")
            except subprocess.CalledProcessError:
                click.echo(
                    "Error: Unable to commit changes. Please check your git repository."
                )
        else:
            click.echo("Commit cancelled.")


if __name__ == "__main__":
    main()
