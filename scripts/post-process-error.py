#!/usr/bin/env python3

import re
import argparse
from collections import defaultdict
from colorama import Fore, Back, Style

def print_log(text = "", color = Fore.LIGHTBLACK_EX):
    """Print colored text to the standard output (gray by default)"""
    print(color + text + Style.RESET_ALL, flush = True)

def process_error_entries(args):
    """Process error entries from raw input files"""

    # Regular expression
    error_pattern = re.compile(
        r"\[Error\]\s+Relative\s+L2-norm\s+=\s+([0-9.eE+-]+)\s+"
        r"RMS\s+norm\s+=\s+([0-9.eE+-]+)\s+"
        r"Relative\s+infinity\s+norm\s+=\s+([0-9.eE+-]+)"
    )
    order_pattern = re.compile(r"\[param\]\s+--order\s+=\s+(\d+)")

    # Get file names
    input_file = str(args.input_file)
    result_file = str(args.result_file)

    # Read the file and parse
    values = defaultdict()
    with open(input_file, 'r') as file:
        for line in file:
            match = order_pattern.search(line)
            if match:
                values["order"] = int(match.group(1))
            match = error_pattern.search(line)
            if match:
                values["Relative L2-norm"] = float(match.group(1))
                values["RMS norm"] = float(match.group(2))
                values["Relative infinity norm"] = float(match.group(3))

    # Print recorded values
    if args.verbose >= 2:
        print_log("\t--- report ---")
        for category, value in values.items():
            print_log(f"{category} = {value}")
        print()

    # Save results into output file
    format_string = "{:>8d}" + "{:>15e}" * 3 + '\n'
    with open(result_file, 'a') as file:
        file.write(format_string.format(
            int(values["order"]),
            float(values["Relative L2-norm"]),
            float(values["RMS norm"]),
            float(values["Relative infinity norm"])
        ))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Post process results for errors.")
    parser.add_argument("--input-file", type=str, help="Name of the input file that contains the raw results.", required=True)
    parser.add_argument("--result-file", type=str, help="Name of the result (updated in 'append' mode).", required=True)
    parser.add_argument("-v", "--verbose", action='count', default=0, help="Increase verbosity level")
    return parser.parse_args()

def print_args(args):
    """Print arguments form parser."""
    print_log("\n\t--- parameters ---", Fore.CYAN)
    for arg, value in vars(args).items():
        print_log(f"<params> {arg} = {value}", Fore.CYAN)
    print_log()

def main():
    """Main function."""
    args = parse_arguments()
    print_args(args) if args.verbose >= 2 else None
    process_error_entries(args)

if __name__ == "__main__":
    main()
