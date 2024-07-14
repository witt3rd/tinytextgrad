import os
from dataclasses import dataclass
from typing import Any, Tuple

from dotenv import load_dotenv
from loguru import logger

from tinytextgrad.llm import call_llm
from tinytextgrad.prompt import Prompt, load_prompt

#

load_dotenv()

package_dir = os.path.dirname(os.path.abspath(__file__))
SYS_PROMPT_DIR = os.path.join(package_dir, "sys_prompts")
if not os.path.exists(SYS_PROMPT_DIR):
    raise ValueError(f"Sys prompt directory {SYS_PROMPT_DIR} does not exist.")

APPLY_GRADIENT_PROMPT = load_prompt(
    name="apply_gradient",
    prompt_dir=SYS_PROMPT_DIR,
)

CLEANUP_TEXT_PROMPT = load_prompt(
    name="cleanup_text",
    prompt_dir=SYS_PROMPT_DIR,
)

PROMPT_LOSS_FN_INSTRUCTIONS = load_prompt(
    name="prompt_loss",
    prompt_dir=SYS_PROMPT_DIR,
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
DEFAULT_MODEL_TEMPERATURE = float(os.getenv("DEFAULT_MODEL_TEMPERATURE", "0.7"))
DEFAULT_MODEL_MAX_TOKENS = int(os.getenv("DEFAULT_MODEL_MAX_TOKENS", "2048"))
DEFAULT_MODEL_TOP_P = float(os.getenv("DEFAULT_MODEL_TOP_P", "0.95"))
DEFAULT_MODEL_FREQUENCY_PENALTY = float(
    os.getenv("DEFAULT_MODEL_FREQUENCY_PENALTY", "0")
)

DEFAULT_EVAL_MODEL = os.getenv("DEFAULT_EVAL_MODEL", "gpt-4o")
DEFAULT_EVAL_MODEL_TEMPERATURE = float(
    os.getenv("DEFAULT_EVALMODEL_TEMPERATURE", "0.7")
)
DEFAULT_EVAL_MODEL_MAX_TOKENS = int(os.getenv("DEFAULT_EVALMODEL_MAX_TOKENS", "2048"))
DEFAULT_EVAL_MODEL_TOP_P = float(os.getenv("DEFAULT_EVAL_MODEL_TOP_P", "0.95"))
DEFAULT_EVAL_MODEL_FREQUENCY_PENALTY = float(
    os.getenv("DEFAULT_EVALMODEL_FREQUENCY_PENALTY", "0")
)


#


@dataclass
class OptimizationResult:
    """
    Represents the result of an optimization.
    """

    variable: "Variable"
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float

    #

    def __str__(self) -> str:
        return self.variable.value

    def __repr__(self) -> str:
        return f"OptimizationResult(variable={self.variable}, model={self.model}, temperature={self.temperature}, max_tokens={self.max_tokens}, top_p={self.top_p}, frequency_penalty={self.frequency_penalty})"

    #

    def to_prompt(
        self,
    ) -> Prompt:
        """
        Create a prompt from an optimization result.
        """
        return Prompt(
            template=self.variable.value,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
        )


class Engine:
    """
    Represents an LLM engine.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty

    def generate(
        self,
        prompt: str,
        prompt_input: str,
    ) -> Any:
        """
        Generate a response from the given prompt and input.
        """
        response = call_llm(
            prompt=prompt,
            prompt_input=prompt_input,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
        )
        return response


class Variable:
    """
    Represents a variable in the optimization process.
    """

    def __init__(
        self,
        value,
        requires_grad=True,
        role_description="",
    ) -> None:
        self.value = value
        self.requires_grad = requires_grad
        self.role_description = role_description
        self.grad = None

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Variable(value={self.value}, requires_grad={self.requires_grad}, role_description={self.role_description})"

    def set_gradient(self, grad: str) -> None:
        """
        Set the gradient of the variable.
        """
        if self.requires_grad:
            self.grad = grad

    def backward(self, application_prompt: str, engine) -> None:
        """
        Applies the gradient to the variable value using an engine.
        """
        if self.requires_grad and self.grad:
            new_value = engine.generate(
                prompt=application_prompt,
                prompt_input=(
                    f"[ORIGINAL TEXT]:\n{self.value}\n\n"
                    + f"[FEEDBACK]:\n{self.grad}\n\n"
                    + "[REVISED TEXT]:\n"
                ),
            )
            self.value = self._clean_text(new_value, engine).strip()
            logger.debug(f"∇ Updated value:\n\n{self.value}")

    def _clean_text(
        self,
        text: str,
        engine: Engine,
    ) -> Any:
        """
        Clean the text using the given engine.
        """
        cleaned_text = engine.generate(
            prompt=CLEANUP_TEXT_PROMPT.render(),
            prompt_input=f"[ORIGINAL TEXT]:\n{text}\n\n[CLEANED TEXT]:\n",
        ).strip()
        return cleaned_text.strip()


class TextLoss:
    """
    Represents a text loss function.
    """

    def __init__(
        self,
        feedback_prompt: str,
        engine: Engine,
    ) -> None:
        self.feedback_prompt = feedback_prompt
        self.engine = engine

    def forward(
        self,
        text: str,
        results: list[Tuple[str, str]],
    ) -> Any:
        """
        Calculate the loss for the given text and results.
        """
        formatted_results = "\n".join(
            [f"Input: {input}\nOutput: {output}" for input, output in results]
        )
        evaluation_input = f"Text:\n{text}\n\nResults:\n{formatted_results}"
        logger.debug(f"∇ Evaluation input:\n\n{evaluation_input}")
        loss = self.engine.generate(self.feedback_prompt, evaluation_input)
        return loss


class TGD:
    """
    Represents the Text Gradient Descent (TGD) algorithm.
    """

    def __init__(
        self,
        variable: Variable,
        model_engine: Engine,
        eval_engine: Engine,
        loss_fn: TextLoss,
        inputs: list[str],
    ) -> None:
        self.variable = variable
        self.model_engine = model_engine
        self.eval_engine = eval_engine
        self.loss_function = loss_fn
        self.inputs = inputs

    def generate_results(self) -> list[Any]:
        """
        Generate results for the inputs.
        """
        results = []
        for _input in self.inputs:
            output = self.model_engine.generate(self.variable.value, _input)
            results.append((_input, output))
        return results

    def step(self) -> None:
        """
        Perform a single step of the optimization process.
        """
        results = self.generate_results()
        loss = self.loss_function.forward(self.variable.value, results)
        logger.debug(f"∇ Loss:\n\n{loss}")
        self.variable.set_gradient(loss)
        self.apply_gradient()

    def apply_gradient(self) -> None:
        """
        Apply the gradient to the variable.
        """
        apply_gradient_prompt = APPLY_GRADIENT_PROMPT.render()
        self.variable.backward(apply_gradient_prompt, self.eval_engine)

    def optimize_text(
        self,
        num_iterations: int = 5,
    ) -> OptimizationResult:
        """
        Optimize the text using the given number of iterations.
        """
        for i in range(num_iterations):
            logger.debug(f"∇ {i+1}. Current prompt:\n\n{self.variable.value}")
            self.step()

        return OptimizationResult(
            variable=self.variable,
            model=self.model_engine.model_name,
            temperature=self.model_engine.temperature,
            max_tokens=self.model_engine.max_tokens,
            top_p=self.model_engine.top_p,
            frequency_penalty=self.model_engine.frequency_penalty,
        )


#


def optimize_prompt(
    initial_prompt: str,
    prompt_inputs: list[str],
    model: str | Engine = DEFAULT_MODEL,
    eval_model: str | Engine = DEFAULT_EVAL_MODEL,
    num_iterations: int = 5,
) -> OptimizationResult:
    """
    Optimize a prompt using the given model and eval model.
    """
    logger.trace(
        f"Optimizing prompt: {initial_prompt}, model: {model}, eval model: {eval_model}, inputs: {prompt_inputs}, num_iterations: {num_iterations}"
    )
    if isinstance(model, str):
        model_engine = Engine(
            model_name=model,
            temperature=DEFAULT_MODEL_TEMPERATURE,
            max_tokens=DEFAULT_MODEL_MAX_TOKENS,
            top_p=DEFAULT_MODEL_TOP_P,
            frequency_penalty=DEFAULT_MODEL_FREQUENCY_PENALTY,
        )
    else:
        model_engine = model

    if isinstance(eval_model, str):
        eval_engine = Engine(
            model_name=eval_model,
            temperature=DEFAULT_EVAL_MODEL_TEMPERATURE,
            max_tokens=DEFAULT_EVAL_MODEL_MAX_TOKENS,
            top_p=DEFAULT_EVAL_MODEL_TOP_P,
            frequency_penalty=DEFAULT_EVAL_MODEL_FREQUENCY_PENALTY,
        )
    else:
        eval_engine = eval_model

    variable = Variable(
        value=initial_prompt,
        role_description="Prompt to optimize",
    )

    loss_fn = TextLoss(
        PROMPT_LOSS_FN_INSTRUCTIONS.render(),
        eval_engine,
    )

    optimizer = TGD(
        variable=variable,
        model_engine=model_engine,
        eval_engine=eval_engine,
        loss_fn=loss_fn,
        inputs=prompt_inputs,
    )

    optimized_text = optimizer.optimize_text(num_iterations=num_iterations)
    return optimized_text
