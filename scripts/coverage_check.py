import subprocess
import sys


def main():
    try:
        # Run the coverage commands
        args = ["coverage", "run", "--source=src", "tests/main_test.py"]
        subprocess.run(args, check=True)
        subprocess.run(["coverage", "report", "-m"], check=True)
    except subprocess.CalledProcessError:
        sys.exit(1)  # Exit with a non-zero code on failure


if __name__ == "__main__":
    main()
