from unittest.mock import Mock

import pytest
from tinytextgrad import TGD, Engine, OptimizationResult, TextLoss, Variable


@pytest.fixture
def engine():
    return Engine(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=2048,
        top_p=0.95,
        frequency_penalty=0,
    )


@pytest.fixture
def variable():
    return Variable(
        "Initial prompt value", requires_grad=True, role_description="Test role"
    )


@pytest.fixture
def text_loss(engine):
    return TextLoss("Feedback prompt", engine)


def test_engine_generate(engine):
    result = engine.generate("Test prompt", "Test input")
    assert isinstance(result, str)
    assert len(result) > 0


def test_variable_set_gradient(variable):
    variable.set_gradient("Test gradient")
    assert variable.grad == "Test gradient"


def test_variable_backward(engine, variable):
    variable.set_gradient("Test gradient")
    original_value = variable.value
    variable.backward("Application prompt", engine)
    assert variable.value != original_value
    assert isinstance(variable.value, str)
    assert len(variable.value) > 0


def test_text_loss_forward(text_loss):
    results = [("Input 1", "Output 1"), ("Input 2", "Output 2")]
    loss = text_loss.forward("Test text", results)
    assert isinstance(loss, str)
    assert len(loss) > 0


def test_tgd_generate_results():
    engine = Engine()
    variable = Variable("Test prompt")
    tgd = TGD(variable, engine, engine, Mock(), ["Input 1", "Input 2"])
    results = tgd.generate_results()
    assert len(results) == 2
    assert all(isinstance(item, tuple) and len(item) == 2 for item in results)
    assert all(
        isinstance(item[0], str) and isinstance(item[1], str) for item in results
    )


def test_tgd_step():
    engine = Engine()
    variable = Variable("Test prompt")
    loss_fn = TextLoss("Feedback prompt", engine)
    tgd = TGD(variable, engine, engine, loss_fn, ["Input 1"])
    original_value = variable.value
    tgd.step()
    assert variable.value != original_value


def test_tgd_optimize_text():
    engine = Engine()
    variable = Variable("Test prompt")
    loss_fn = TextLoss("Feedback prompt", engine)
    tgd = TGD(variable, engine, engine, loss_fn, ["Input 1"])
    result = tgd.optimize_text(num_iterations=3)
    assert isinstance(result, OptimizationResult)
    assert result.variable == variable
    assert result.model == "gpt-3.5-turbo"


if __name__ == "__main__":
    pytest.main()
