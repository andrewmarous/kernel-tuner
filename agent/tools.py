import os
import re
import io
import csv
import subprocess

class KernelTuner:
    def __init__(self, source_path="kernel.cu"):
        self.source_path = os.path.join("../kernels", source_path)
        self.output_bin = "./matmul_bench"

    def _parse_ncu_csv(self, raw_output: str, requested_metrics: list) -> dict:
        """Safely extracts metric values from NCU's messy CSV output."""
        parsed_data = {}

        # Initialize our dictionary with None so we know if a metric failed to load
        for metric in requested_metrics:
            parsed_data[metric] = None

        # Read line by line to skip the NVIDIA warnings and find the actual CSV headers
        lines = raw_output.strip().split('\n')
        csv_start_idx = 0
        for i, line in enumerate(lines):
            # The actual CSV data always starts with the header row beginning with "ID" or "Host Name"
            if line.startswith('"ID"') or line.startswith('"Host Name"'):
                csv_start_idx = i
                break

        # Grab only the valid CSV portion
        valid_csv_str = '\n'.join(lines[csv_start_idx:])

        # Parse using Python's built-in CSV reader
        reader = csv.DictReader(io.StringIO(valid_csv_str))

        # NCU outputs one row per kernel launch.
        # We will grab the metrics from the first kernel execution row.
        for row in reader:
            metric_name = row.get("Metric Name")
            metric_value = row.get("Metric Value")

            if metric_name in parsed_data:
                try:
                    # Convert the string value to a float for the agent to do math on
                    # NCU sometimes includes commas in large numbers, so we remove them
                    parsed_data[metric_name] = float(metric_value.replace(',', ''))
                except ValueError:
                    parsed_data[metric_name] = metric_value # Fallback if it's not a number

        return parsed_data

    def update_params(self, block_x, block_y):
        """Edits the MATMUL_BLOCK_SIZE macros in the source file."""
        with open(self.source_path, 'r') as f:
            content = f.read()

        # Update MATMUL_BLOCK_SIZE_X
        content = re.sub(
            r"(#define MATMUL_BLOCK_SIZE_X )\d+",
            rf"\1{block_x}",
            content
        )
        # Update MATMUL_BLOCK_SIZE_Y
        content = re.sub(
            r"(#define MATMUL_BLOCK_SIZE_Y )\d+",
            rf"\1{block_y}",
            content
        )

        with open(self.source_path, 'w') as f:
            f.write(content)
        print(f"Updated source to: BlockX={block_x}, BlockY={block_y}")

    def compile(self):
        """Compiles with nvcc."""
        # Compile command (adjust architecture flag as needed, e.g., -arch=sm_80)
        compile_cmd = ["nvcc", self.source_path, "-o", self.output_bin]

        try:
            result = subprocess.run(compile_cmd, check=True, capture_output=True)
            result.check_returncode()
        except subprocess.CalledProcessError as e:
            return f"Error during compilation/execution: {e.stderr}"

    def profile_kernel(self) -> dict:
        metrics_to_collect = [
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_active"
        ]

        metrics_str = ",".join(metrics_to_collect)

        ncu_cmd = [
            "ncu",
            "--csv",
            "--page", "details",
            "--metrics", metrics_str,
            self.output_bin
        ]

        print(f"Executing: {' '.join(ncu_cmd)}")

        try:
            result = subprocess.run(ncu_cmd, check=True, capture_output=True, text=True)
            raw_csv_output = result.stdout

            return self._parse_ncu_csv(raw_csv_output, metrics_to_collect)

        except subprocess.CalledProcessError as e:
            print(f"NCU Profiling failed: {e.stderr}")
            return {"error": str(e.stderr)}
        except FileNotFoundError:
            return {"error": "ncu command not found"}

# --- Example Usage for the Agent ---
# tuner = KernelTuner()
# tuner.update_params(32, 16)
# output = tuner.compile_and_run()
# print(output)
