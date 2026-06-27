class GeminiRequestError(RuntimeError):
    pass


def generate_with_gemini(selection, model_name, content, model_factory):
    try:
        model = model_factory(model_name)
        return model.generate_content(content)
    except Exception as exc:
        error_text = str(exc)
        if selection.api_key:
            error_text = error_text.replace(selection.api_key, "[redacted]")
        raise GeminiRequestError(
            f"Gemini request failed using {selection.key_name}: {error_text}"
        ) from None
