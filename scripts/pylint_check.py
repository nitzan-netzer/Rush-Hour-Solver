import os
import subprocess


def main():
    for file in os.listdir("src"):
        if file.endswith(".py"):
            args = ["pylint", f"src/{file}"]
            try:
                subprocess.run(args, check=True)
            except subprocess.CalledProcessError:
                print(f"pylint check failed for {file}")


if __name__ == "__main__":
    main()
