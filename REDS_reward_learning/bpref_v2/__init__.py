try:
    import rich.traceback
    from rich import print

    rich.traceback.install()
except ImportError:
    print(f"ImportError")
    pass
