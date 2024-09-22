#from source.languagemodel import LanguageModel
from source.utilities import (
    convert_tokens_to_songdata,
    convert_songdata_to_notesequence,
    convert_songdata_to_pianoroll,
    convert_notesequence_to_wave,
    convert_notesequence_to_midi
)
from datasets import load_dataset
import random
import os
import fire
import numpy as np
import hashlib
import sqlite3
from PIL import Image
import tqdm


# Get two random token sequences.
#dataset_size = len(dataset["train"])
#index1 = random.randint(0, dataset_size)
#index2 = random.randint(0, dataset_size)
#token_sequence1 = dataset["train"][index1]["text"]
#token_sequence2 = dataset["train"][index2]["text"]

image_mode = "RGB"

def main(command):
    available_commands = ["fill", "test"]
    if command not in available_commands:
        raise ValueError(f"Command {command} not in available commands {available_commands}")
    
    if command == "fill":
        fill_database()
    elif command == "test":
        test_database()
    else:
        raise ValueError(f"Command {command} not implemented.")
    #test()
    #fill_database()
    

def test_database():

    # Load the dataset.
    dataset_id = "TristanBehrens/bach_garland_2024-1M"
    dataset = load_dataset(dataset_id)
    print(dataset)

    # Get the dataset size.
    dataset_size = len(dataset["train"])

    # Get a random token sequence.
    index = random.randint(0, dataset_size)
    token_sequence = dataset["train"][index]["text"]
    score = find_similar_samples(token_sequence)
    print(f"Score: {score}")


def fill_database():

    # Load the dataset.
    dataset_id = "TristanBehrens/bach_garland_2024-1M"
    dataset = load_dataset(dataset_id)
    print(dataset)

    # Get the dataset size.
    dataset_size = len(dataset["train"])

    # Now let us fill a file based relational database with the sample indices their images, an their pitch ranges.
    # Use the hash as the primary key. 
    # Print a warning if the hash already exists in the database.

    # Connect to the database.
    # Delete the database if it already exists.
    if os.path.exists("database.db"):
        print("Database already exists. Do you want to delete it? (y/n)")
        answer = input()
        if answer != "y":
            print("Aborting.")
            exit()
        os.remove("database.db")
        print("Database deleted.")
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    # Create the table. If it already exists, do nothing.
    try:
        cursor.execute("CREATE TABLE samples (hash TEXT PRIMARY KEY, image BLOB, image_width INTEGER, image_height INTEGER)")
        print("Table created.")
    except sqlite3.OperationalError:
        print("Table already exists.")
        pass

    # Insert the samples.
    # Flush every 100 samples.
    hash_collisions_counter = 0
    insertion_counter = 0
    progress_bar = tqdm.tqdm(total=dataset_size)
    for index in range(dataset_size):
        token_sequence = dataset["train"][index]["text"]
        image, min_pitch, max_pitch, hash = convert_token_sequence(token_sequence)
        image = image.convert(image_mode)
        try: 
            cursor.execute("INSERT INTO samples (hash, image, image_width, image_height) VALUES (?, ?, ?, ?)", (hash, image.tobytes(), image.size[0], image.size[1]))
            insertion_counter += 1
        except sqlite3.IntegrityError as e:
            #print(hash)
            #print(e)
            #exit()
            #print(f"Hash {hash} already exists in the database.")
            hash_collisions_counter += 1
        if index % 100 == 0:
            conn.commit()
        # Update the progress bar. Use the current index. Also include the hash collisions and the insertion counter.
        progress_bar.update(1)
        progress_bar.set_postfix({"hash collisions": hash_collisions_counter, "insertions": insertion_counter})
    print(f"Hash collisions: {hash_collisions_counter:_}/{dataset_size:_}")

    # Commit the remaining samples.
    conn.commit()


def find_similar_samples(token_sequence, top_n=10):

    if not os.path.exists("database.db"):
        return "UNKNOWN"
    
    # Connect to the database.
    connection = sqlite3.connect("database.db")
    cursor = connection.cursor()

    # Convert the token sequence to an image.
    image, min_pitch, max_pitch, hash = convert_token_sequence(token_sequence)

    # Query the database for similar samples.
    cursor.execute("SELECT * FROM samples")
    rows = cursor.fetchall()

    similar_samples = []

    # Compute the similarity scores.
    progress_bar = tqdm.tqdm(total=len(rows))
    for row in rows:
        row_image = Image.frombytes(image_mode, (row[2], row[3]), row[1])
        overlap_score, range_score, overlap = compute_scores(image, row_image)
        # Add the similarity score to the list.
        similar_samples.append((overlap_score, range_score, row[0], overlap))
        # Sort by overlap score ascending.
        similar_samples.sort(key=lambda x: x[0], reverse=True)
        # Keep only the top n similar samples.
        similar_samples = similar_samples[:top_n]
        progress_bar.update(1)

    # Close the database connection.
    connection.close()

    # Save the similar samples and the original sample. 
    output_path = "output/similar_samples"
    if not os.path.exists(output_path):
        os.makedirs(output_path)    

    # Save the original sample.
    image.save(os.path.join(output_path, "original.png"))
    for i, sample in enumerate(similar_samples):
        hash = sample[2]
        connection = sqlite3.connect("database.db")
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM samples WHERE hash = ?", (hash,))
        row = cursor.fetchone()
        image_bytes = row[1]
        image_width = row[2]
        image_height = row[3]
        image = Image.frombytes(image_mode, (image_width, image_height), image_bytes)
        overlap = sample[3]
        colorize_overlap(image, overlap)
        image.save(os.path.join(output_path, f"similar_sample_{i}.png"))
        connection.close()

    # Print the similar samples scores.
    print("Similar samples:")
    for i, sample in enumerate(similar_samples):
        print(f"Sample {i}: Overlap score: {sample[0]:.2f} Range score: {sample[1]:.2f} Hash: {sample[2]}")

    # Return the score.
    return similar_samples[0][0]


def colorize_overlap(image, overlap):
    # Use the overlap to change the background of the first overlap columns form black to dark red.
    for j in range(overlap):
        for y in range(image.size[1]):
            color = image.getpixel((j, y))
            if color == (0, 0, 0):
                image.putpixel((j, y), (64, 0, 0))



def match_for_similarity(token_sequence, image):
    assert isinstance(token_sequence, str), f"Token sequence must be a string. Got {type(token_sequence)}"
    assert isinstance(image, Image.Image), f"Image must be a PIL image. Got {type(image)}"

    image1, min_pitch1, max_pitch1, hash1 = convert_token_sequence(token_sequence)
    image2 = image
    #print(image1.size, type(image1))
    #print(image2.size, type(image2))
    return compute_scores(image1, image2)


def convert_token_sequence(token_sequence):

    # Convert to song data.
    song_data = convert_tokens_to_songdata(token_sequence)
    song_data["bpm"] = 120

    # Convert to piano roll.
    image, min_pitch, max_pitch = convert_songdata_to_pianoroll(song_data, monochrome=True, return_min_max=True, scale=1)

    # Convert to hash use the image.
    hash = hashlib.sha256()
    hash.update(image.tobytes())
    hash_hex = hash.hexdigest()

    return image, min_pitch, max_pitch, hash_hex

    # Save the image.
    #if not os.path.exists(os.path.dirname(image_path)):
    #    os.makedirs(os.path.dirname(image_path))
    #image.save(image_path)


def match_token_sequences(token_sequence_1, token_sequence_2):
    assert isinstance(token_sequence_1, str), f"Token sequence 1 must be a string. Got {type(token_sequence_1)}"
    assert isinstance(token_sequence_2, str), f"Token sequence 2 must be a string. Got {type(token_sequence_2)}"

    image1, min_pitch1, max_pitch1, hash1 = convert_token_sequence(token_sequence_1)
    image2, min_pitch2, max_pitch2, hash2 = convert_token_sequence(token_sequence_2)

    return compute_scores(image1, image2)


def compute_scores(image1, image2, min_pitch1=None, max_pitch1=None, min_pitch2=None, max_pitch2=None):
    
    # Should be PIL images.
    assert isinstance(image1, Image.Image), f"Image 1 must be a PIL image. Got {type(image1)}"
    assert isinstance(image2, Image.Image), f"Image 2 must be a PIL image. Got {type(image2)}"

    # Should have the same mode.
    assert image1.mode == image2.mode, f"Image 1 and 2 must have the same mode. Got {image1.mode} and {image2.mode}"

    # Compute the overlap score.
    matching_columns = 0
    max_columns = max(image1.size[0], image2.size[0])
    for columns in range(max_columns):
        # Get the slices.
        slice1 = image1.crop((0, 0, columns + 1, image1.size[1]))
        slice2 = image2.crop((0, 0, columns + 1, image2.size[1]))
        assert slice1.size[0] == columns + 1, f"Slice 1 must have width {columns + 1}. Got {slice1.size[0]}"
        assert slice2.size[0] == columns + 1, f"Slice 2 must have width {columns + 1}. Got {slice2.size[0]}"
        assert slice1.size[1] == image1.size[1], f"Slice 1 must have the same height as image 1. Got {slice1.size[1]} and {image1.size[1]}"
        assert slice2.size[1] == image2.size[1], f"Slice 2 must have the same height as image 2. Got {slice2.size[1]} and {image2.size[1]}"

        # Remove black border at the top and bottom.
        slice1 = remove_black_rows(slice1)
        slice2 = remove_black_rows(slice2)

        assert slice2.size[0] == slice1.size[0], f"Slice 1 and 2 must have the same width. Got {slice1.size[0]} and {slice2.size[0]}"
        assert slice1.size[0] == columns + 1, f"Slice 1 must have the same height as the columns. Got {slice1.size[1]} and {columns + 1}"

        # Abort if the heights are different.
        if slice1.size[1] != slice2.size[1]:
            break

        # Compare the slices.
        if slice1.tobytes() == slice2.tobytes():
            matching_columns += 1

    overlap_score = matching_columns / max_columns

    # Compute the range scores.
    if any([min_pitch1 is None, max_pitch1 is None, min_pitch2 is None, max_pitch2 is None]):
        range_score = 0
    else:
        total_min_pitch = min(min_pitch1, min_pitch2)
        total_max_pitch = max(max_pitch1, max_pitch2)
        count = 0
        for i in range(total_min_pitch, total_max_pitch):
            if i >= min_pitch1 and i <= max_pitch1 and i >= min_pitch2 and i <= max_pitch2:
                count += 1
        range_score = count / (total_max_pitch - total_min_pitch)

    return overlap_score, range_score, matching_columns


def remove_black_rows(image):
    # Convert image to numpy array
    img_array = np.array(image)

    # Check for black rows (all values are 0)
    # Sum along the color channels, if the sum is 0, it's a black row
    row_sums = img_array.sum(axis=(1, 2))

    # Find the first non-black row from the top
    top = 0
    while top < len(row_sums) and row_sums[top] == 0:
        top += 1

    # Find the first non-black row from the bottom
    bottom = len(row_sums) - 1
    while bottom > top and row_sums[bottom] == 0:
        bottom -= 1

    # Crop the image to the region that has non-black rows
    cropped_image = image.crop((0, top, image.width, bottom + 1))

    return cropped_image


def test():

    # Load the dataset.
    dataset_id = "TristanBehrens/bach_garland_2024-1M"
    dataset = load_dataset(dataset_id)
    print(dataset)

    # Get the dataset size.
    dataset_size = len(dataset["train"])

    # Get two random token sequences.
    index1 = random.randint(0, dataset_size)
    index2 = random.randint(0, dataset_size)
    token_sequence1 = dataset["train"][index1]["text"]
    token_sequence2 = dataset["train"][index2]["text"]

    # Convert the token sequences to images.
    image1, min_pitch1, max_pitch1, hash1 = convert_token_sequence(token_sequence1)
    image2, min_pitch2, max_pitch2, hash2 = convert_token_sequence(token_sequence2)
    print(f"Hash1: {hash1}")
    print(f"Hash2: {hash2}")

    overlap_score, range_score = match_token_sequences(token_sequence1, token_sequence2)
    print(f"Overlap score: {overlap_score} Range score: {range_score}")

    overlap_score, range_score = match_token_sequences(token_sequence1, token_sequence1)
    print(f"Overlap score: {overlap_score} Range score: {range_score}")

    overlap_score, range_score = match_token_sequences(token_sequence2, token_sequence2)
    print(f"Overlap score: {overlap_score} Range score: {range_score}")



if __name__ == "__main__":
   fire.Fire(main)