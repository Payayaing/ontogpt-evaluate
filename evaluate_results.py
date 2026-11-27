import os
import json
import csv
from collections import Counter

def fetch_file_locations(model_folder: str) -> tuple[str, str, str, str, str, str, str, str, str, str]: # Return assessed samples that have attached predicted triples, time taken to perform inference, and extraction success. Success is determined whether OntoGPT parsed the result without throwing an error.
    # Model results directory are the results stored in the corresponding model folder.
    return (f"dataset/predictions/{model_folder}/ont_1_movie_results.jsonl", f"dataset/predictions/{model_folder}/ont_2_music_results.jsonl",
        f"dataset/predictions/{model_folder}/ont_3_sport_results.jsonl", f"dataset/predictions/{model_folder}/ont_4_book_results.jsonl", 
        f"dataset/predictions/{model_folder}/ont_5_military_results.jsonl", f"dataset/predictions/{model_folder}/ont_6_computer_results.jsonl",
        f"dataset/predictions/{model_folder}/ont_7_space_results.jsonl", f"dataset/predictions/{model_folder}/ont_8_politics_results.jsonl",
        f"dataset/predictions/{model_folder}/ont_9_nature_results.jsonl", f"dataset/predictions/{model_folder}/ont_10_culture_results.jsonl")

def unpack_jsonl(directory: str) -> list[dict]: # Unpack JSONL results of one class.
    class_results = []
    with open(directory, "r", encoding="utf-8") as j:
        for line in j:
            class_results.append(json.loads(line))
    return class_results

def export_csv(results: list[dict], directory: str): # Export results as a CSV file.
    with open(directory, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

def evaluate_model(model: str):
    files = fetch_file_locations(model)
    class_results = [] # Store numerical results from each class. Each model will have generalised performance across all classes as well as performance for each class.
    for file in files:
        class_name = os.path.basename(file).removesuffix("_results.jsonl")
        class_predictions = unpack_jsonl(file)
        class_result = evaluate_class(samples=class_predictions)
        class_result["model"] = model
        class_result["class"] = class_name
        class_results.append(class_result)
    return class_results

def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

def evaluate_class(samples: list[dict]) -> dict:
    sample_results = [ # Get results from each triple independently.
        evaluate_single_sample(sample)
        for sample in samples
    ]
    class_totals = { # Sum all the result values across all samples in the class.
        key: sum(result[key] for result in sample_results)
        for key in sample_results[0]
    }
    triple_extraction_ratio = safe_divide( # Indicates whether a model tends to overextract (>1) or underextract (<1)
        class_totals["total_predicted_triples"],
        class_totals["total_ground_truth_triples"]
    )
    # Using the formula precision = TP / (TP + FP)
    precision = safe_divide( # Share of predicted positives that are actually positive
        class_totals["correct_complete_triples"],
        class_totals["total_predicted_triples"]
    )
    # Using the formula recall = TP / (TP + FN)
    # FN (false negatives) = total ground truth triples - correct complete triples
    recall = safe_divide( # Share of actual positives that are correctly identified
        class_totals["correct_complete_triples"],
        class_totals["total_ground_truth_triples"]
    )
    # Using the formula f1 = 2 x ((precision x recall) / (precision + recall))
    f1_score = safe_divide( # Harmonic mean of precision and recall
        2 * precision * recall,
        precision + recall
    )
    predicate_precision = safe_divide(
        class_totals["matching_predicates_occurrences"],
        class_totals["total_predicted_triples"]
    )
    predicate_recall = safe_divide(
        class_totals["matching_predicates_occurrences"],
        class_totals["total_ground_truth_triples"]
    )
    predicate_f1_score = safe_divide(
        2 * predicate_precision * predicate_recall,
        predicate_precision + predicate_recall
    )
    
    extraction_success_rate = class_totals["extraction_success"] / len(sample_results) # How many samples got through the OntoGPT parser
    non_empty_extraction_rate = class_totals["non_empty_extraction"] / len(sample_results) # How many samples had 1 or more triples successfully extracted
    
    mean_elapsed_time = safe_divide(
        class_totals["elapsed_time"],
        len(sample_results)
    )
    
    class_totals["triple_extraction_ratio"] = triple_extraction_ratio
    class_totals["precision"] = precision
    class_totals["recall"] = recall
    class_totals["f1_score"] = f1_score
    class_totals["predicate_precision"] = predicate_precision
    class_totals["predicate_recall"] = predicate_recall
    class_totals["predicate_f1_score"] = predicate_f1_score
    class_totals["extraction_success_rate"] = extraction_success_rate
    class_totals["non_empty_extraction_rate"] = non_empty_extraction_rate
    class_totals["total_samples"] = len(sample_results)
    class_totals["mean_elapsed_time"] = mean_elapsed_time
    class_totals["total_elapsed_time"] = class_totals.pop("elapsed_time") # Making the key more descriptive of what the columns actually contain
    
    return class_totals

def normalise(triple: dict) -> tuple[str, str, str]: # Format subject, predicate, and object within each triple to be consistent for accurate and truthful comparisons.
    # Remove all spaces and lower case. Any other differences are deemed model extraction errors.
    return tuple(
        "".join(triple[key].split()).lower()
        for key in ("sub", "rel", "obj")
    )

def evaluate_single_sample(sample: dict) -> dict: # Compute and return quantitative metrics for a provided sample.
    # Each sample contains id, sent, triples, predicted_triples, elapsed_time, and status keys. triples and predicted_triples have identical structures, 
    # containing "sub", "rel", and "obj" keys for subject, predicate (relation), object.
    # Things that should be evaluated per sample:
    # Number of successful extractions, number of failed extractions, number of successful extractions that results in an empty predicted triples list
    # How many predicate relations from ground truth are present in predictions
    # Correct predicate wrong entities, correct predicate correct entities
    extraction_failure, empty_extraction = 0, 0
    if sample["status"] == "failed": # Where OntoGPT threw an error for this extraction, resulting in no triples returned.
        extraction_failure = 1
    elif len(sample["predicted_triples"]) == 0: # OntoGPT did not throw an error for this extraction, but no triples were extracted.
        empty_extraction = 1
    ground_truth_triples = {normalise(triple) for triple in sample["triples"]} 
    predicted_triples = {normalise(triple) for triple in sample["predicted_triples"]}

    ground_truth_predicates = [triple[1] for triple in ground_truth_triples] # Extract predicates from both ground truths and predicted triples.
    predicted_predicates = [triple[1] for triple in predicted_triples]
    ground_truth_set, predicted_set = set(ground_truth_predicates), set(predicted_predicates) # Comparing unique predicates using sets.
    ground_truth_count, predicted_count = Counter(ground_truth_predicates), Counter(predicted_predicates) # Comparing predicate frequencies using Counter.
    
    matching_predicates = ground_truth_set & predicted_set
    missing_predicates = ground_truth_set - predicted_set
    unexpected_predicates = predicted_set - ground_truth_set
    matching_count = ground_truth_count & predicted_count
    missing_count = ground_truth_count - predicted_count
    unexpected_count = predicted_count - ground_truth_count
    correct_triples = len(ground_truth_triples & predicted_triples)
    num_matching, num_missing, num_unexpected = sum(matching_count.values()), sum(missing_count.values()), sum(unexpected_count.values())
    correct_predicate_wrong_entities = num_matching - correct_triples
    incorrect_triples = len(predicted_triples - ground_truth_triples)
    missing_triples = len(ground_truth_triples - predicted_triples)
    
    return {
        "total_ground_truth_triples": len(ground_truth_triples),
        "total_predicted_triples": len(predicted_triples),
        "unique_matching_predicates": len(matching_predicates),
        "unique_missing_predicates": len(missing_predicates),
        "unique_unexpected_predicates": len(unexpected_predicates),
        "matching_predicates_occurrences": num_matching,
        "missing_predicates_occurrences": num_missing,
        "unexpected_predicates_occurrences": num_unexpected,
        "correct_complete_triples": correct_triples,
        "correct_predicate_wrong_entities": correct_predicate_wrong_entities,
        "incorrect_triples": incorrect_triples,
        "missing_triples": missing_triples,
        "extraction_success": 1 - extraction_failure, # 0 if it was not a success, 1 if it was a success
        "extraction_failure": extraction_failure,
        "non_empty_extraction": 1 - extraction_failure - empty_extraction,
        "empty_extraction": empty_extraction,
        "elapsed_time": sample["elapsed_time"]
    }

all_class_results = []
for model in ("ollama_qwen3_8b", "ollama_deepseek-r1_8b", "ollama_mistral-nemo_12b", "ollama_gemma3_4b"): # Calculate results for all models.
    print(f"Evaluating model results: {model}")
    all_class_results.extend(evaluate_model(model))
os.makedirs("evaluation_results/", exist_ok=True)
export_csv(all_class_results, "evaluation_results/class_statistics.csv")