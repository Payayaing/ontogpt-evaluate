import json

def fetch_file_locations() -> list[tuple[str, str, str, str, str]]:
    return [
        ("wikidata_tekgen/test/ont_1_movie_test.jsonl", "wikidata_tekgen/ground_truth/ont_1_movie_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_1_movie.txt", 
            "dataset/test/ont_1_movie_verified.jsonl", "dataset/ground_truth/ont_1_movie_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_2_music_test.jsonl", "wikidata_tekgen/ground_truth/ont_2_music_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_2_music.txt", 
            "dataset/test/ont_2_music_verified.jsonl", "dataset/ground_truth/ont_2_music_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_3_sport_test.jsonl", "wikidata_tekgen/ground_truth/ont_3_sport_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_3_sport.txt", 
            "dataset/test/ont_3_sport_verified.jsonl", "dataset/ground_truth/ont_3_sport_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_4_book_test.jsonl", "wikidata_tekgen/ground_truth/ont_4_book_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_4_book.txt", 
            "dataset/test/ont_4_book_verified.jsonl", "dataset/ground_truth/ont_4_book_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_5_military_test.jsonl", "wikidata_tekgen/ground_truth/ont_5_military_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_5_military.txt", 
            "dataset/test/ont_5_military_verified.jsonl", "dataset/ground_truth/ont_5_military_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_6_computer_test.jsonl", "wikidata_tekgen/ground_truth/ont_6_computer_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_6_computer.txt", 
            "dataset/test/ont_6_computer_verified.jsonl", "dataset/ground_truth/ont_6_computer_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_7_space_test.jsonl", "wikidata_tekgen/ground_truth/ont_7_space_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_7_space.txt", 
            "dataset/test/ont_7_space_verified.jsonl", "dataset/ground_truth/ont_7_space_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_8_politics_test.jsonl", "wikidata_tekgen/ground_truth/ont_8_politics_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_8_politics.txt", 
            "dataset/test/ont_8_politics_verified.jsonl", "dataset/ground_truth/ont_8_politics_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_9_nature_test.jsonl", "wikidata_tekgen/ground_truth/ont_9_nature_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_9_nature.txt", 
            "dataset/test/ont_9_nature_verified.jsonl", "dataset/ground_truth/ont_9_nature_ground_truth_verified.jsonl"),
        ("wikidata_tekgen/test/ont_10_culture_test.jsonl", "wikidata_tekgen/ground_truth/ont_10_culture_ground_truth.jsonl", 
            "wikidata_tekgen/manually_verified_sentences/selected_ont_10_culture.txt", 
            "dataset/test/ont_10_culture_verified.jsonl", "dataset/ground_truth/ont_10_culture_verified.jsonl"),
    ]

def fetch_test_sentences(sentence_location) -> list[dict]:
    sentences = []
    with open(sentence_location, "r", encoding="utf-8") as j:
        for line in j:
            sentences.append(json.loads(line))
    return sentences

def fetch_ground_truths(truth_location) -> list[dict]: # acts as an alias of fetch_test_sentences, has the exact same logic but technically different purpose
    return fetch_test_sentences(sentence_location=truth_location)

def fetch_verified_ids(id_location) -> list[str]:
    verified_ids = []
    with open(id_location, "r", encoding="utf-8") as j:
        for line in j:
            verified_ids.append(line.strip()) # some ids has a \n appended on the end which need to be removed
    return verified_ids

def filter_manually_verified(sentence_location, truth_location, id_location) -> list[dict]: # filtering from both test sentences and ground truths
    sentences = fetch_test_sentences(sentence_location)
    ground_truths = fetch_ground_truths(truth_location)
    verified_ids = fetch_verified_ids(id_location)
    verified_sentences = []
    verified_ground_truths = []
    for i, line in enumerate(sentences): # line is a dictionary containing 'id' and 'sent'
        if line.get('id') in verified_ids: # sentence is verified, so add it to the verified list for both
            verified_sentences.append(line)
            verified_ground_truths.append(ground_truths[i])
    return verified_sentences, verified_ground_truths

def export_verified_sentences_jsonl(verified_sentences, export_sentence_location) -> None:
    with open(export_sentence_location, "w", encoding="utf-8") as f:
        for sentence in verified_sentences:
            f.write(json.dumps(sentence, ensure_ascii=False) + "\n")

def export_ground_truths_jsonl(verified_ground_truths, export_truth_location) -> None: # acts as an alias of export_verified_sentences_jsonl, has the exact same logic but technically different purpose
    export_verified_sentences_jsonl(verified_sentences=verified_ground_truths, export_sentence_location=export_truth_location)

def get_verified_sentences(sentence_location, truth_location, id_location, export_sentence_location, export_truth_location):
    verified_sentences, verified_ground_truths = filter_manually_verified(sentence_location, truth_location, id_location)
    export_verified_sentences_jsonl(verified_sentences, export_sentence_location)
    export_ground_truths_jsonl(verified_ground_truths, export_truth_location)

if __name__ == "__main__":
    files = fetch_file_locations()
    for sentence_location, truth_location, id_location, export_sentence_location, export_truth_location in files:
        get_verified_sentences(sentence_location, truth_location, id_location, export_sentence_location, export_truth_location)