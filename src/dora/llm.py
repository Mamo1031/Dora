"""Local LLM implementation using Ollama and LangChain."""


from langchain_community.llms import Ollama


class LocalLLM:
    """Wrapper class for local LLM using Ollama."""

    def __init__(self, model_name: str = "llama3.2") -> None:
        """Initialize the local LLM.

        Parameters
        ----------
        model_name : str, optional
            Name of the Ollama model to use, by default "llama3.2"

        Raises
        ------
        ConnectionError
            If Ollama is not running or the model is not available
        """
        self.model_name = model_name
        try:
            self.llm = Ollama(model=model_name)
        except Exception as e:
            error_msg = (
                f"Failed to connect to Ollama or model '{model_name}' is not available. "
                f"Make sure Ollama is running and the model is pulled: ollama pull {model_name}"
            )
            raise ConnectionError(error_msg) from e

    def invoke(self, prompt: str) -> str:
        """Generate a response from the LLM.

        Parameters
        ----------
        prompt : str
            Input prompt for the LLM

        Returns
        -------
        str
            Generated response from the LLM

        Raises
        ------
        RuntimeError
            If the LLM fails to generate a response
        """
        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            error_msg = f"Failed to generate response: {e}"
            raise RuntimeError(error_msg) from e
        else:
            return response
