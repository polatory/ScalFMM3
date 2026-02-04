#!/usr/bin/env python3

import re
import argparse
from collections import defaultdict
from colorama import Fore, Back, Style

def print_log(text = "", color = Fore.LIGHTBLACK_EX):
    """Print colored text to the standard output (gray by default)"""
    print(color + text + Style.RESET_ALL, flush = True)

def process_time_entries(args):
    """Process time entries from raw input files"""

    # Regular expressions
    time_pattern = re.compile(r"\[time\]\[([^\]]+)\]\s+:\s+([0-9.eE+-]+)")
    order_pattern = re.compile(r"\[param\]\s+--order\s+=\s+(\d+)")
    tree_height_pattern = re.compile(r"\[param\]\s+--tree-height\s+=\s+(\d+)")
    nb_particles_pattern = re.compile(r"\[(generate|file)\]\[nb-particles\]\s+:\s+(\d+)")

    # Get file names
    input_file = str(args.input_file)
    result_file = str(args.result_file)

    entries = defaultdict(list)

    # Read the file and parse
    values = defaultdict()
    with open(input_file, 'r') as file:
        for line in file:
            match = order_pattern.search(line)
            if match:
                values["order"] = int(match.group(1))
            match = time_pattern.search(line)
            if match:
                category = match.group(1)
                value = float(match.group(2))
                entries[category].append(value)
            match = nb_particles_pattern.search(line)
            if match:
                values["nb-particles"] = int(match.group(2))
            match = tree_height_pattern.search(line)
            if match:
                values["tree-height"] = int(match.group(1))

    # Compute averages
    averages = {category: sum(times) / len(times) for category, times in entries.items()}

    for category, avg in sorted(averages.items()):
        values[category] = float(avg)

    # Print recorded values
    if args.verbose >= 2:
        print_log("\t--- report ---")
        for category, value in values.items():
            print_log(f"{category} = {value}")
        print()

    # Save results into output file
    format_string = "{:>8d}" + "{:>4d}" + "{:>4d}" + "{:>15e}" * 10 + '\n'
    with open(result_file, 'a') as file:
        file.write(format_string.format(
            int(values["nb-particles"]),
            int(values["order"]),
            int(values["tree-height"]),
            float(values["fmm"]),
            float(values["bottom pass"]),
            float(values["upward pass"]),
            float(values["transfer pass"]),
            float(values["downward pass"]),
            float(values["cell_to_leaf pass"]),
            float(values["direct pass"]),
            float(values["far time"]),
            float(values["near time"]),
            float(values["ratio time"])
        ))

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Post process results for time.")
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
    process_time_entries(args)

if __name__ == "__main__":
    main()
