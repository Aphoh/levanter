#!/usr/bin/env python3

import json
import subprocess
import sys


def delete_preempted_tpus():
    """
    Lists Google Cloud TPUs, identifies preempted ones, and deletes them.
    """
    list_command = ["gcloud", "alpha", "compute", "tpus", "list", "--format=json"]

    print("Running 'gcloud alpha compute tpus list --format=json'...")

    try:
        # Run the gcloud list command
        result = subprocess.run(list_command, capture_output=True, text=True, check=True)
        tpus_json_output = result.stdout

        # Parse the JSON output
        tpu_list = json.loads(tpus_json_output)

    except FileNotFoundError:
        print(
            "Error: 'gcloud' command not found. Please ensure Google Cloud SDK is installed and in your PATH.",
            file=sys.stderr,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running gcloud list command: {e}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON output from gcloud command.", file=sys.stderr)
        print(f"Raw output:\n{tpus_json_output}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during listing: {e}", file=sys.stderr)
        sys.exit(1)

    preempted_tpus = [tpu for tpu in tpu_list if tpu.get("state") == "PREEMPTED"]

    if not preempted_tpus:
        print("No preempted TPUs found.")
        return

    print(f"Found {len(preempted_tpus)} preempted TPU(s).")

    for tpu in preempted_tpus:
        full_name = tpu.get("name")
        if not full_name:
            print(f"Skipping TPU with no 'name' field: {tpu}", file=sys.stderr)
            continue

        # The 'name' field is in the format: projects/PROJECT_ID/locations/ZONE/nodes/TPU_NAME
        # We need to extract ZONE and TPU_NAME for the delete command.
        try:
            parts = full_name.split("/")
            if len(parts) >= 6 and parts[4] == "nodes":
                zone = parts[3]
                tpu_name = parts[5]
            else:
                print(
                    f"Warning: Could not parse zone and name from unexpected full name format: {full_name}. Skipping.",
                    file=sys.stderr,
                )
                continue  # Skip this TPU if name format is unexpected

        except IndexError:
            print(
                f"Warning: Could not parse zone/name from name field parts for '{full_name}'. Skipping.",
                file=sys.stderr,
            )
            continue

        print(f"Attempting to delete TPU '{tpu_name}' in zone '{zone}'...")

        delete_command = [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            "--zone",
            zone,
            tpu_name,
            "--quiet",
            "--async",
        ]  # Added --quiet to prevent prompts

        try:
            # Run the delete command
            subprocess.run(delete_command, capture_output=True, text=True, check=True)
            print(f"Successfully deleted '{tpu_name}' in zone '{zone}'.")
            # Optional: print delete output if needed
            # print(f"Delete stdout:\n{delete_result.stdout}")
            # print(f"Delete stderr:\n{delete_result.stderr}")

        except subprocess.CalledProcessError as e:
            print(f"Error deleting TPU '{tpu_name}' in zone '{zone}': {e}", file=sys.stderr)
            print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred during deletion of '{tpu_name}': {e}", file=sys.stderr)

    print("Deletion process finished.")


if __name__ == "__main__":
    delete_preempted_tpus()
