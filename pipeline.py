import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# -------------------------------
# Configuration
# -------------------------------

RAW_DIR = Path('raw_experiment_data')
STEP1_DIR = Path('step1_output')
STEP2_DIR = Path('step2_output')

RAW_DIR.mkdir(exist_ok=True)
STEP1_DIR.mkdir(exist_ok=True)
STEP2_DIR.mkdir(exist_ok=True)


# -------------------------------
# Utility functions
# -------------------------------

def load_json(file_path: Path) -> List[Dict]:
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: Path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


# -------------------------------
# Step 1: Extract Relevant Data
# -------------------------------

class Step1Extractor:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.failed_dir = output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)

    async def watch_and_process(self):
        logging.info("Step1Extractor is watching for files.")
        while True:
            for file in self.input_dir.glob("*.json"):
                await self.process_file(file)
            await asyncio.sleep(1)  # Poll every second

    async def process_file(self, file_path: Path):
        try:
            data = load_json(file_path)
            # Extract minimal info: cell_type, environment, cell_response
            relevant_data = [{
                'cell_type': entry['cell_type']['name'],
                'environment': entry['environment']['name'],
                'cell_response': entry['cell_response']
            } for entry in data]

            out_file = self.output_dir / file_path.name
            save_json(relevant_data, out_file)
            logging.info(f"Step1: Processed and saved {out_file}")

            # Remove processed file
            os.remove(file_path)
        except Exception as e:
            failed_file = self.failed_dir / file_path.name
            file_path.replace(failed_file)
            logging.error(f"Step1 failed for {file_path}: {e}")


# -------------------------------
# Step 2: Validate Hypothesis
# -------------------------------

class Step2HypothesisValidator:
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir

    async def watch_and_process(self):
        logging.info("Step2HypothesisValidator is watching for files.")
        while True:
            for file in self.input_dir.glob("*.json"):
                await self.process_file(file)
            await asyncio.sleep(4)

    async def process_file(self, file_path: Path):
        try:
            data = load_json(file_path)
            neuron_responses = [d['cell_response'] for d in data 
                                if d['cell_type'] == 'Neuron' and d['environment'] == 'In vivo']
            other_responses = [d['cell_response'] for d in data 
                               if not (d['cell_type'] == 'Neuron' and d['environment'] == 'In vivo')]

            hypothesis_valid = False
            if neuron_responses and other_responses:
                avg_neuron = sum(neuron_responses) / len(neuron_responses)
                avg_others = sum(other_responses) / len(other_responses)
                hypothesis_valid = avg_neuron > avg_others

            out_file = self.output_dir / file_path.name
            save_json([{'hypothesis_valid': hypothesis_valid}], out_file)
            logging.info(f"Step2: Processed {file_path.name}, hypothesis valid: {hypothesis_valid}")

            # Remove processed file
            os.remove(file_path)
        except Exception as e:
            logging.error(f"Step2 failed for {file_path}: {e}")


# -------------------------------
# Step 3: Aggregate and Print Summary
# -------------------------------

class Step3Aggregator:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.total = 0
        self.valid_count = 0

    async def watch_and_process(self):
        logging.info("Step3Aggregator is watching for files.")
        while True:
            for file in self.input_dir.glob("*.json"):
                await self.process_file(file)
            await asyncio.sleep(1)

    async def process_file(self, file_path: Path):
        try:
            data = load_json(file_path)
            self.total += 1
            if data[0].get('hypothesis_valid'):
                self.valid_count += 1
            percent = (self.valid_count / self.total) * 100
            logging.info(f"Hypothesis is true for: {percent:.2f}% across experiments.")

            # Remove processed file
            os.remove(file_path)
        except Exception as e:
            logging.error(f"Step3 failed for {file_path}: {e}")


# -------------------------------
# Main runner
# -------------------------------

async def main():
    step1 = Step1Extractor(RAW_DIR, STEP1_DIR)
    step2 = Step2HypothesisValidator(STEP1_DIR, STEP2_DIR)
    step3 = Step3Aggregator(STEP2_DIR)

    await asyncio.gather(
        step1.watch_and_process(),
        step2.watch_and_process(),
        step3.watch_and_process()
    )

if __name__ == "__main__":
    asyncio.run(main())
