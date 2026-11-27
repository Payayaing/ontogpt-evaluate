import os
import json
import random
import logging
import time
import argparse
from ontogpt.engines.spires_engine import SPIRESEngine

logging.basicConfig(level=logging.INFO)

def fetch_file_locations() -> list[tuple[str, str]]: # Returns directory of ground truths and the corresponding ontology respectively.
    return [
        ("dataset/ground_truth/ont_1_movie_ground_truth_verified.jsonl", "dataset/ontologies/1_movie_ontology.yaml"),
        ("dataset/ground_truth/ont_2_music_ground_truth_verified.jsonl", "dataset/ontologies/2_music_ontology.yaml"),
        ("dataset/ground_truth/ont_3_sport_ground_truth_verified.jsonl", "dataset/ontologies/3_sport_ontology.yaml"),
        ("dataset/ground_truth/ont_4_book_ground_truth_verified.jsonl", "dataset/ontologies/4_book_ontology.yaml"),
        ("dataset/ground_truth/ont_5_military_ground_truth_verified.jsonl", "dataset/ontologies/5_military_ontology.yaml"),
        ("dataset/ground_truth/ont_6_computer_ground_truth_verified.jsonl", "dataset/ontologies/6_computer_ontology.yaml"),
        ("dataset/ground_truth/ont_7_space_ground_truth_verified.jsonl", "dataset/ontologies/7_space_ontology.yaml"),
        ("dataset/ground_truth/ont_8_politics_ground_truth_verified.jsonl", "dataset/ontologies/8_politics_ontology.yaml"),
        ("dataset/ground_truth/ont_9_nature_ground_truth_verified.jsonl", "dataset/ontologies/9_nature_ontology.yaml"),
        ("dataset/ground_truth/ont_10_culture_ground_truth_verified.jsonl", "dataset/ontologies/10_culture_ontology.yaml"),
    ]

def get_random_samples(domain_location: str, n: int, seed=24843322) -> list[dict]: # Returns a dictionary of n random samples from the specified knowledge domain.
    sentences = []
    with open(domain_location, "r", encoding="utf-8") as j: # Domain location should be /dataset/ground_truth/{}.jsonl file. List of dictionaries with id, sent, and triples keys.
        for line in j:
            sentences.append(json.loads(line))
    rnd = random.Random(seed) # Set randomness seed to ensure experiment reproducibility. Default seed is my student ID number.
    # Randomness seed also makes sure that all models evaluate the same sentences for a valid comparison.
    if n > len(sentences) or n <= 0:
        logging.warning(f"Requested {n} samples from {domain_location}, which contains {len(sentences)} samples.") # Log an error if there are too many samples
        n = len(sentences) # Set sample size to class population
    return rnd.sample(sentences, n) # Select n unique samples without replacement, guaranteeing that all sentences are unique.

def initialise_spires_engine(template: str, model: str) -> SPIRESEngine: # Return a SPIRESEngine with the specified ontology (template) and model. For this experiment, all models are Ollama local models.
    return SPIRESEngine(
        template=template,
        model=model,
        temperature=1.0 # Although a lower temperature reduces non-determinism in model response, it caused a response quality drop 
    )

def perform_extraction(model: str, num_samples: int) -> None:
    domain_and_ontology_locations = fetch_file_locations()
    start = time.perf_counter()
    for domain_location, ontology_location in domain_and_ontology_locations:
        print(f"Accessing class: {domain_location}")
        ground_truth_samples = get_random_samples(domain_location, n=num_samples)
        class_results = perform_extraction_on_class(ground_truth_samples, ontology_location, model)
        store_result(model, domain_location, class_results)
    end = time.perf_counter()
    time_taken = end - start
    print(f"Total experiment time: {time_taken:.2f} seconds")

def perform_extraction_on_class(samples: list[dict], ontology_location: str, model: str) -> list[dict]:
    engine = initialise_spires_engine(ontology_location, model)
    class_results = []
    for sample in samples: # Loop through all obtained samples and perform per-sample extraction. Time how long all extractions take for additional info.
        sample_result = perform_extraction_on_sample(sample, engine)
        class_results.append(sample_result)
    return class_results

def perform_extraction_on_sample(sample: dict, engine: SPIRESEngine) -> dict:
    print(f"Accessing sample: {sample["id"]}")
    print(f"Sent: {sample["sent"]}")
    raw_llm_output = ""
    start_sample = time.perf_counter()
    try:
        result = engine.extract_from_text(sample["sent"]) # Perform extraction on that particular sample. Time how long this extraction takes for additional info.
        end_sample = time.perf_counter()
        raw_llm_output = result.raw_completion_output
        extracted_object = (
            result.extracted_object.model_dump(mode="json")
            if result.extracted_object is not None
            else {}
        )
        extracted_triples = [ # Convert triples to Text2KGBench ground truth structure. They label subject, predicate, and object slightly differently which needs to be accommodated for.
            {
                "sub": triple["subject"],
                "rel": triple["predicate"].replace("_", " "),
                "obj": triple["object"]
            }
            for triple in (extracted_object.get("triples") or [])
        ]
        elapsed_time = end_sample - start_sample
        print(f"Raw LLM Output: {raw_llm_output}")
        print(f"Extracted Triples: {extracted_triples}")
        print(f"Total elapsed time for sample {sample["id"]} is {elapsed_time:.2f} seconds\n")
        return attach_result(sample, extracted_triples, raw_llm_output, elapsed_time, status="success")
    except Exception as e:
        elapsed_time = time.perf_counter() - start_sample
        logging.exception(f"Extraction failed for {sample["id"]}: {e}\n")
        return attach_result(sample, result=[], raw_llm_output=raw_llm_output, elapsed_time=elapsed_time, status="failed") # Failed extraction.

def attach_result(sample: dict, result: list[dict], raw_llm_output: str, elapsed_time: float, status: str) -> dict: # Attaches predicted triples and elapsed time to a copy of the sample.
    sample_result = sample.copy() # Creates a shallow copy of the original sample. The values of the dictionaries themselves are not being modified, only keys are being added, so this does not pose an issue.
    sample_result["predicted_triples"] = result
    sample_result["raw_llm_output"] = raw_llm_output
    sample_result["elapsed_time"] = elapsed_time
    sample_result["status"] = status
    return sample_result

def store_result(model: str, domain_location: str, results: list[dict]) -> None:
    model_directory = model.replace("/", "_").replace(":", "_") # Replaces ollama/qwen3:8b with ollama_qwen3_8b
    directory = os.path.join("dataset", "predictions", model_directory)
    os.makedirs(directory, exist_ok=True)
    domain_name = os.path.basename(domain_location).removesuffix("_ground_truth_verified.jsonl")
    export_location = os.path.join(directory, f"{domain_name}_results.jsonl")
    with open(export_location, "w", encoding="utf-8") as f:
        for sample in results:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

parser = argparse.ArgumentParser(description="Perform OntoGPT triples extraction.")
parser.add_argument("-m", "--model", type=str, default="ollama/qwen3:8b", help="The LLM that OntoGPT should utilise for extraction.") # First argument is to specify what model should be used. All models will iterate through all templates so an argument for that specifically is excluded.
# List of models that have been used in this experiment:
# ollama/mistral-nemo:12b
# ollama/llama3.1:8b
# ollama/qwen3:8b
# ollama/phi4-mini:latest
# ollama/deepseek-r1:8b
parser.add_argument("-n", "--num-samples", type=int, default=10, help="How many samples from each class should be used for evaluation?") # Second argument is to specify how many samples of each class should be used. Default is 10.
args = parser.parse_args()
print(f"Using model: {args.model}, selecting {args.num_samples} samples across each class")
perform_extraction(args.model, args.num_samples)
