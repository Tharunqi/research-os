import os

from openai import OpenAI


def generate_from_excerpts(
    instructions: str,
    user_input: str,
) -> str:
    """Generate a grounded tool response from retrieved excerpts."""

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")

    response = client.responses.create(
        model=model,
        instructions=(
            instructions
            + "\n\nFormat mathematical expressions with TeX delimiters: "
            "use $...$ inline and $$...$$ on separate lines for display "
            "equations. Do not leave raw TeX commands un-delimited."
        ),
        input=user_input,
    )

    return response.output_text.strip()
