import sys
import argparse

def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def extract_last_time(input_file: str):
    with open(input_file) as fin:
        lines = fin.readlines()
        last_col = lines[-1].split()[-1]
        if is_number(last_col):
            return float(f"{float(last_col):.6f}")
        else:
            return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(f"{sys.argv[0]}")
    parser.add_argument("input_file", type=str, help="file containing execution time (last line)")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(-1)
    args = parser.parse_args()

    input_file = args.input_file

    time = extract_last_time(input_file)

    print(time)
