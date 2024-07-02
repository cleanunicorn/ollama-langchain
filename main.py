import subprocess
import click
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate

def get_git_diff():
    try:
        result = subprocess.run(["git", "diff", "--cached"], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return "Error: Unable to get git diff. Make sure you're in a git repository and have staged changes."

def generate_commit_message(diff):
    llm = Ollama(model="llama2")
    
    prompt = PromptTemplate(
        input_variables=["diff"],
        template="""
        Based on the following git diff, generate a commit message in the style of conventional commits.
        The message should have a type, an optional scope, and a description.
        Types include: feat, fix, docs, style, refactor, perf, test, chore.
        
        Git Diff:
        {diff}
        
        Commit Message:
        """
    )
    
    full_prompt = prompt.format(diff=diff)
    response = llm(full_prompt)
    return response.strip()

@click.command()
@click.option('--preview', is_flag=True, help='Preview the commit message without committing')
def main(preview):
    """Generate a commit message based on staged changes using Ollama and LangChain."""
    diff = get_git_diff()
    
    if not diff or diff.startswith("Error"):
        click.echo(diff or "No changes detected.")
        return
    
    commit_message = generate_commit_message(diff)
    
    if preview:
        click.echo("Generated commit message (preview mode):")
        click.echo(commit_message)
    else:
        try:
            subprocess.run(["git", "commit", "-m", commit_message], check=True)
            click.echo("Changes committed with the generated message:")
            click.echo(commit_message)
        except subprocess.CalledProcessError:
            click.echo("Error: Unable to commit changes. Please check your git repository.")

if __name__ == '__main__':
    main()