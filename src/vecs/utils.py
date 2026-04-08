import re


def slugify(text: str) -> str:
    """Convert text to a safe filename slug."""
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")[:100] or "untitled"
