from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Tuple

from loguru import logger

from tinytextgrad.llm import call_llm

#

APPLY_GRADIENT_PROMPT = dedent("""
    Generate [REVISED TEXT] from the [ORIGINAL TEXT] based on the [FEEDBACK].
    Do not include examples unless they were part of the [ORIGINAL TEXT].
    The [REVISED TEXT] should reflect the [FEEDBACK].
    The [REVISED TEXT] should not contain explanations or meta-commentary.
    """).strip()

CLEANUP_TEXT_PROMPT = dedent("""
    Given the following [ORIGINAL TEXT], generate [CLEANED TEXT] by
    remove any meta-commentary or explanations about the text itself.
    Leave all other [ORIGINAL TEXT] unchanged.
    """).strip()

#


@dataclass
class OptimizationResult:
    variable: "Variable"
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float


class Engine:
    def __init__(
        self,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        frequency_penalty=0,
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

    def set_gradient(self, grad: str) -> None:
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
        cleaned_text = engine.generate(
            prompt=CLEANUP_TEXT_PROMPT,
            prompt_input=f"[ORIGINAL TEXT]:\n{text}\n\n[CLEANED TEXT]:\n",
        ).strip()
        return cleaned_text.strip()


class TextLoss:
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
        formatted_results = "\n".join(
            [f"Input: {input}\nOutput: {output}" for input, output in results]
        )
        evaluation_input = f"Text:\n{text}\n\nResults:\n{formatted_results}"
        logger.debug(f"∇ Evaluation input:\n\n{evaluation_input}")
        loss = self.engine.generate(self.feedback_prompt, evaluation_input)
        return loss


class TGD:
    def __init__(
        self,
        variable: Variable,
        model_engine: Engine,
        eval_engine: Engine,
        loss_fn: TextLoss,
        inputs: list[str],
    ):
        self.variable = variable
        self.model_engine = model_engine
        self.eval_engine = eval_engine
        self.loss_function = loss_fn
        self.inputs = inputs

    def generate_results(self) -> list[Any]:
        results = []
        for _input in self.inputs:
            output = self.model_engine.generate(self.variable.value, _input)
            results.append((_input, output))
        return results

    def step(self):
        results = self.generate_results()
        loss = self.loss_function.forward(self.variable.value, results)
        logger.debug(f"∇ Loss:\n\n{loss}")
        self.variable.set_gradient(loss)
        self.apply_gradient()

    def apply_gradient(self) -> None:
        apply_gradient_prompt = APPLY_GRADIENT_PROMPT
        self.variable.backward(apply_gradient_prompt, self.eval_engine)

    def optimize_text(
        self,
        num_iterations: int = 5,
    ) -> OptimizationResult:
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
