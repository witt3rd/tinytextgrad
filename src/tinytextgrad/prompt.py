import os
from dataclasses import dataclass

import yaml
from dotenv import load_dotenv
from jinja2 import Template

from tinytextgrad.llm import call_llm

#

load_dotenv()

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_MODEL_TEMPERATURE = float(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0.7"))
DEFAULT_MODEL_MAX_TOKENS = int(os.getenv("DEFAULT_MODEL_MAX_TOKENS", "2048"))
DEFAULT_MODEL_TOP_P = float(os.getenv("DEFAULT_MODEL_TOP_P", "0.95"))
DEFAULT_MODEL_FREQUENCY_PENALTY = float(
    os.getenv("DEFAULT_MODEL_FREQUENCY_PENALTY", "0")
)

PROMPT_DIR = os.getenv("PROMPT_DIR", "prompts")
os.makedirs(PROMPT_DIR, exist_ok=True)

#


def parse_frontmatter(
    content: str,
) -> tuple[dict, str]:
    """
    Parse frontmatter from a string content.

    Parameters
    ----------
    content : str
        The input string containing potential frontmatter.

    Returns
    -------
    tuple[dict, str]
        A tuple containing two elements:
        - dict: The parsed frontmatter as a dictionary.
        - str: The remaining content after frontmatter.

    Notes
    -----
    Frontmatter should be enclosed in triple dashes
    (---) at the beginning of the content. If no
    frontmatter is found, an empty dictionary is
    returned along with the original content.
    """
    metadata = {}
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            metadata = yaml.safe_load(content[3:end])
            if not isinstance(metadata, dict):
                metadata = {}
            content = content[end + 3 :].strip()
    return metadata, content


#


@dataclass
class Prompt:
    """
    A class representing a prompt for the LLM with
    parameters and Jinja2 templating support.
    """

    template: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_MODEL_TEMPERATURE
    max_tokens: int = DEFAULT_MODEL_MAX_TOKENS
    top_p: float = DEFAULT_MODEL_TOP_P
    frequency_penalty: float = DEFAULT_MODEL_FREQUENCY_PENALTY
    response_format_type: str = "text"  # or "json_object"

    #

    def __str__(self) -> str:
        return self.template

    def __repr__(self) -> str:
        return f"Prompt(template={self.template}, model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p}, frequency_penalty={self.frequency_penalty}, response_format_type={self.response_format_type})"

    #

    @classmethod
    def from_markdown_file(
        cls,
        file_path: str,
    ) -> "Prompt":
        """
        Create a prompt from a markdown file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            metadata, prompt_text = parse_frontmatter(f.read())
        if not prompt_text:
            raise ValueError("Markdown file does not contain content.")
        return cls(
            model=str(metadata.get("model", DEFAULT_MODEL)),
            template=prompt_text,
            temperature=float(
                metadata.get(
                    "temperature",
                    DEFAULT_MODEL_TEMPERATURE,
                )
            ),
            max_tokens=int(
                metadata.get(
                    "max_tokens",
                    DEFAULT_MODEL_MAX_TOKENS,
                )
            ),
            top_p=float(metadata.get("top_p", DEFAULT_MODEL_TOP_P)),
            frequency_penalty=float(
                metadata.get(
                    "frequency_penalty",
                    DEFAULT_MODEL_FREQUENCY_PENALTY,
                )
            ),
            response_format_type=str(metadata.get("response_format_type", "text")),
        )

    #

    def save(
        self,
        name: str,
        prompt_dir: str = PROMPT_DIR,
    ) -> str:
        """
        Save the prompt to a file in the given directory.
        """
        full_path = os.path.join(prompt_dir, f"{name}.md")
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(self.template)
        return full_path

    def render(self, **kwargs) -> str:
        """
        Render the prompt template with the given variables.
        """
        template = Template(self.template)
        return template.render(**kwargs)

    def call_llm(self, prompt_input: str, **kwargs) -> str:
        """
        Call the LLM with the given prompt input.
        """
        return call_llm(
            prompt=self.render(**kwargs),
            prompt_input=prompt_input,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            response_format_type=self.response_format_type,
        )


def load_prompt(
    name: str,
    prompt_dir: str = PROMPT_DIR,
) -> Prompt:
    """
    Load a prompt from a file in the given directory.
    """
    full_path = os.path.join(prompt_dir, f"{name}.md")
    return Prompt.from_markdown_file(file_path=full_path)
