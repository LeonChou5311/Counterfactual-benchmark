from sys import stdout

def print_block(content: str, title="", num_marks=20):
    upper= ("=" * num_marks) + title + ("=" * num_marks)
    bottom = "=" * len(upper)
    stdout.write(
        "\n" +
        upper + "\n" +
        "| %s " % (content) + "\n" +
        bottom + "\n"
    )
    stdout.flush()
