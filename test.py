"""
Script for testing your decoder on sample data.
If you pass the tests here, your code probably won't crash on Gradescope.

Usage: python3 test.py

WARNING: We recommend you don't modify this file.
"""

from typing import List, NamedTuple, Optional, Tuple

import logging
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import argparse


def first_line(file_path):
    # Return first line of file as string, without trailing newline
    with open(file_path) as f:
        return f.readline().rstrip("\r\n")


class RunResult(NamedTuple):
    stdout: str = ""
    stderr: str = ""
    elapsed_secs: float = 0.0
    crash_reason: Optional[str] = None  # None means no crash


def run_decode_cli(
    executable_path: str,
    ciphertext: str,
    has_breakpoint: bool,
    command_prefix: Optional[List[str]] = None,
    timeout_secs: Optional[float] = None,
    test_name: Optional[str] = None,
    debug: bool = False,
) -> RunResult:
    """
    return: output, elapsed_secs, crash_reason
    An output of None indicates a crash.
    """
    if command_prefix is None:
        command_prefix = []

    if not os.path.exists(executable_path):
        logging.error("decode-cli does not exist")
        return RunResult(crash_reason="decode-cli does not exist")

    # Explicitly call python on windows
    if platform.system() == "Windows":
        command_prefix.append(sys.executable)
    else:
        # Otherwise ensure executable can be executed
        subprocess.call(["chmod", "+x", executable_path])

    executable_dir = os.path.dirname(
        executable_path)  # foo/bar/decode-cli -> foo/bar
    executable_file = os.path.basename(
        executable_path)  # foo/bar/decode-cli -> decode-cli

    crash_reason = None
    start_time_secs = time.monotonic()

    # Based off of https://stackoverflow.com/a/36955420/1337463
    with subprocess.Popen(
            cwd=executable_dir,
            args=command_prefix + [
                f"./{executable_file}",
                ciphertext,
                str(has_breakpoint),
                test_name or "test",
                str(debug),
            ],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="UTF-8",
    ) as process:
        try:
            process_outputs = process.communicate(timeout=timeout_secs)
        except subprocess.TimeoutExpired:
            # Terminate with extreme prejudice
            # (ref: https://stackoverflow.com/a/4047975/1337463)
            logging.error("decode-cli timed out")
            logging.error("killing TLEd process group...")
            os.killpg(process.pid, signal.SIGKILL)
            logging.error("killed TLEd process group")

            crash_reason = "time limit exceeded"
            process_outputs = process.communicate()

        stdout, stderr = (o.strip("\r\n") for o in process_outputs)

        if crash_reason is None and process.returncode != 0:
            logging.error("decode-cli failed")
            logging.error(f"return code: {process.returncode}")
            logging.error("STDERR:")
            logging.error(stderr)

            crash_reason = "generic crash"

    end_time_secs = time.monotonic()
    elapsed_secs = end_time_secs - start_time_secs

    if crash_reason is None and len(stdout) != len(ciphertext):
        logging.error("Decoded output must have same length as ciphertext")
        logging.error(f"Output     length: {len(stdout)}")
        logging.error(f"Ciphertext length: {len(ciphertext)}")
        logging.error(f"Output:")
        logging.error(f"\"{stdout}\"")
        crash_reason = "Decoded output has invalid length"

    return RunResult(
        stdout=stdout,
        stderr=stderr,
        elapsed_secs=elapsed_secs,
        crash_reason=crash_reason,
    )


def count_matches(a: str, b: str) -> int:
    """Returns the number of locations where the two strings are equal."""
    assert len(a) == len(b)
    return sum(int(i == j) for i, j in zip(a, b))


def fail_if_crash(rr: RunResult):
    if rr.crash_reason is None:
        return

    print()
    print()
    print("!!! ERROR !!!")
    print(f"Reason: {rr.crash_reason}")
    print("Your code seems to have errors.",
          "Please fix them and then rerun this test.")
    exit(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp", action="store_true")
    parser.add_argument("--no_bp", action="store_true")
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--length", nargs='+', type=int)
    parser.add_argument("--select", nargs='+')
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    assert args.length is None or len(args.length) == 2
    if not args.bp and not args.no_bp:
        args.bp = True
        args.no_bp = True
    return args


def main():
    args = parse_args()

    from src.encode import assert_clean

    logging.basicConfig(format="%(levelname)s - %(message)s")

    executable_path = "./decode-cli"
    dummy_text = "the quick brown fox jumped over the lazy dog."
    assert_clean(dummy_text)

    test_case_path = "data/test_cases"
    total_match_no_bp = 0
    total_compared_no_bp = 0
    total_match_bp = 0
    total_compared_bp = 0
    test_names = []
    for test_file in os.listdir(test_case_path):
        test_name, ext = os.path.splitext(test_file)
        if ext != '.out':
            continue
        if args.select is not None and test_name not in args.select:
            continue
        if args.short != (test_name[-1] == 's'):
            continue
        plaintext = first_line(os.path.join(test_case_path, test_file))
        if args.length is not None and not (args.length[0] <= len(plaintext) <= args.length[1]):
            continue
        test_names.append(test_name)

    print(f"{len(test_names)} tests selected")
    print()
    for i, test_name in enumerate(sorted(test_names)):
        plaintext = first_line(os.path.join(test_case_path, test_name + '.out'))
        print(f"Test #{i+1}: {test_name}")
        print(f"Length: {len(plaintext)}")

        if args.no_bp:
            ciphertext = first_line(os.path.join(test_case_path, test_name + '.in'))
            print("Running no breakpoint test...")
            res = run_decode_cli(executable_path, ciphertext, False, test_name=test_name, debug=args.debug)
            fail_if_crash(res)
            num_match = count_matches(plaintext, res.stdout)
            total_match_no_bp += num_match
            total_compared_no_bp += len(plaintext)
            print(
                f"Score (no breakpoint): {num_match} out of {len(plaintext)}"
            )
            print(f"Elapsed secs (no breakpoint): {res.elapsed_secs}")
            print()

        if args.bp:
            ciphertext_with_breakpoint = first_line(os.path.join(test_case_path, test_name + '_bp.in'))
            print("Running breakpoint test...")
            res = run_decode_cli(executable_path, ciphertext_with_breakpoint, True, test_name=f"{test_name}_bp", debug=args.debug)
            fail_if_crash(res)
            num_match = count_matches(plaintext, res.stdout)
            total_match_bp += num_match
            total_compared_bp += len(plaintext)
            print(
                f"Score (breakpoint): {num_match} out of {len(plaintext)}"
            )
            print(f"Elapsed secs (breakpoint): {res.elapsed_secs}")
            print()

        print("SUCCESS")
        print(f"decode-cli ran succesfully on {test_name}.")
        print()
        print()

    total_match = total_match_bp + total_match_no_bp
    total_compared = total_compared_bp + total_compared_no_bp
    print(f"Total score: {total_match}/{total_compared} = {total_match / total_compared}")
    if args.bp and args.no_bp:
        print(f"No breakpoint score: {total_match_no_bp}/{total_compared_no_bp} = {total_match_no_bp / total_compared_no_bp}")
        print(f"Breakpoint score: {total_match_bp}/{total_compared_bp} = {total_match_bp / total_compared_bp}")
    print()

    print("Checking that you are not hardcoding inputs...")
    res = run_decode_cli(executable_path, dummy_text, False)
    fail_if_crash(res)
    count_matches(dummy_text, res.stdout)
    print("DONE")



if __name__ == "__main__":
    main()
