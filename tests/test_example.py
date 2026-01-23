"""Example test to verify pytest is working."""


def test_example() -> None:
    """Simple example test."""
    assert True


def test_import() -> None:
    """Test that the main package can be imported."""
    from video_generation import __version__

    assert __version__ == "0.1.0"
