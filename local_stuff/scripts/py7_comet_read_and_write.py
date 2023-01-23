import os
import sys
import subprocess
import numpy as np


def main():
    if len(sys.argv) != 2:
        print(F"python3 {sys.argv[0]} <input.ta>")
        exit(-1)

    input_txt = sys.argv[1]
    with open(input_txt) as fin:
        lines = fin.readlines();
        # test
        print(F"lines: {lines}")

        last_line = lines[-1];
        # test
        print(F"last_line: {last_line}")


        columns = last_line.split(',')
        columns = columns[:-1]  # remove the last '\n'
        # test
        print(F"columns: {columns}")

        zero_end = -1
        for i in range(len(columns)):
            end_i = len(columns) - i - 1
            if columns[end_i] != '0':
                zero_end = end_i
                break
        if zero_end == -1:
            print("Error: output all zeros")
            exit()
        # test
        print(F"zero_end: {zero_end}")
        for i in range(zero_end + 1):
            print(columns[i])


if __name__ == "__main__":
    main()
