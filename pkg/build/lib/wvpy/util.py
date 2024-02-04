import re


def escape_ansi(line) -> str:
    """https://stackoverflow.com/a/38662876"""
    ansi_escape_regexp = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape_regexp.sub("", str(line))
