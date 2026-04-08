from vecs.utils import slugify


def test_slugify_normal():
    assert slugify("Hello World") == "hello-world"


def test_slugify_special_chars():
    assert slugify("What's up? (2024)") == "whats-up-2024"


def test_slugify_empty_string():
    """Empty input falls back to 'untitled' instead of empty string."""
    assert slugify("") == "untitled"


def test_slugify_only_special_chars():
    """Input with only special chars falls back to 'untitled'."""
    assert slugify("!!!@@@###") == "untitled"


def test_slugify_long_title():
    """Long titles are truncated to 100 chars."""
    long_title = "a" * 200
    result = slugify(long_title)
    assert len(result) <= 100


def test_slugify_unicode():
    """Unicode that becomes empty after stripping gets fallback."""
    assert slugify("\u2603\u2764") == "untitled"


def test_slugify_underscores():
    """Underscores are converted to hyphens."""
    assert slugify("snake_case_title") == "snake-case-title"
