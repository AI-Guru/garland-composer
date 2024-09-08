import copy
import note_seq
from PIL import Image
import tempfile


def convert_tokens_to_songdata(tokens):

    if isinstance(tokens, str):
        tokens = tokens.split()

    song_data = {}

    song_data["tracks"] = []

    current_track_index = 0
    current_timestep = 0
    for token in tokens:
        if token == "GARLAND_START":
            pass
        elif token == "BAR_START":
            if current_track_index == len(song_data["tracks"]):
                song_data["tracks"] += [{"bars": [], "instrument": "0"}]
            bar_data = {"notes": []}
            song_data["tracks"][current_track_index]["bars"] += [bar_data]
            current_timestep = 0
        elif token.startswith("INST="):
            instrument = token.split("=")[1]
            song_data["tracks"][current_track_index]["instrument"] = instrument
        elif token.startswith("DENSITY="):
            pass
        elif token.startswith("NOTE_ON="):
            note_pitch = int(token.split("=")[1])
            note_data = {
                "note": note_pitch,
                "start": current_timestep,
                "end": current_timestep,
                "veloctiy": 80
            }
            song_data["tracks"][current_track_index]["bars"][-1]["notes"] += [note_data]
            pass
        elif token.startswith("TIME_DELTA="):
            current_timestep += int(token.split("=")[1])
        elif token.startswith("NOTE_OFF="):
            note_pitch = int(token.split("=")[1])
            for note_data in song_data["tracks"][current_track_index]["bars"][-1]["notes"]:
                if note_data["note"] == note_pitch and note_data["start"] == note_data["end"]:
                    note_data["end"] = current_timestep
                    break
            pass
        elif token == "BAR_END":
            current_track_index += 1
        elif token == "NEXT":
            current_track_index = 0
        elif token == "GARLAND_END":
            pass
        elif token == "[PAD]":
            pass
        elif token == "[EOS]":
            pass
        else:
            raise Exception(f"Unknown token: {token}")
    
    assert isinstance(song_data, dict)
    return song_data


def convert_songdata_to_notesequence(song_data:dict, quantize_steps_per_quarter=8, remove_disabled_tracks=True):

    assert isinstance(song_data, dict), f"Invalid song data type: {type(song_data)}"

    # Clone the song data.
    song_data = copy.deepcopy(song_data)

    # Sort the tracks by instrument.
    assert "tracks" in song_data, f"Invalid song data: {song_data.keys()}"
    tracks = sorted(song_data["tracks"], key=lambda t: t["instrument"])
    song_data["tracks"] = tracks

    # Remove tracks that are not enabled.
    if remove_disabled_tracks:
        song_data["tracks"] = [t for t in song_data["tracks"] if t.get("enabled", True)]

    # Create an empy note sequence.
    note_sequence = note_seq.protobuf.music_pb2.NoteSequence()

    # Add the tempo.
    bpm = song_data["bpm"] if "bpm" in song_data else 120
    note_sequence.tempos.add().qpm = bpm

    # Compute some lengths.
    step_length_seconds = 60.0 / bpm / quantize_steps_per_quarter
    bar_length_seconds = 4 * step_length_seconds * quantize_steps_per_quarter

    # Get the instruments.
    instruments = list(set([t["instrument"] for t in song_data["tracks"]]))

    # Add the tracks.
    for track_index, track_data in enumerate(song_data["tracks"]):
        instrument = track_data["instrument"]
        for bar_index, bar_data in enumerate(track_data["bars"]):
            bar_start_time = bar_index * bar_length_seconds
            for note_data in bar_data["notes"]:
                assert "note" in note_data
                assert "start" in note_data
                assert "end" in note_data
                note = note_sequence.notes.add()
                #note.instrument = instrument TODO
                note.pitch = note_data["note"]
                note.start_time = note_data["start"] * step_length_seconds + bar_start_time
                note.end_time = note_data["end"] * step_length_seconds + bar_start_time
                if "velocity" in note_data:
                    note.velocity = note_data["velocity"]
                else:
                    note.velocity = 80
                note.instrument = track_index
                if instrument == "drums":
                    note.is_drum = True
                else:
                    note.is_drum = False
                    note.program = int(instrument)

    return note_sequence


def convert_songdata_to_pianoroll(song_data):

    # The bars are 4/4 and the quantization is 8 steps per quarter, aka 32 steps per bar.
    # We will render a grid. The height is 64 pixels. The width is 32 pixels per bar

    # Create a new image.
    lengths = [len(track["bars"]) for track in song_data["tracks"]]
    if lengths == []:
        return None
    assert len(set(lengths)) == 1, f"Unequal number of bars: {lengths}"
    num_bars = lengths[0]

    # Get the note extremes.
    min_note = 128
    max_note = 0
    for track_data in song_data["tracks"]:
        for bar_data in track_data["bars"]:
            for note_data in bar_data["notes"]:
                min_note = min(min_note, note_data["note"])
                max_note = max(max_note, note_data["note"])

    # The width depends on the bars.
    width = 32 * num_bars
    
    # The width depends on the notes.
    height = 1 + max_note - min_note

    # Create the image.
    image = Image.new("RGB", (width, height), (14, 17, 23))

    # Define some colors.
    colors = [
        (255, 75, 75),
    ]

    # Draw the grid.
    for track_index, track_data in enumerate(song_data["tracks"]):
        color = colors[track_index % len(colors)]
        for bar_index, bar_data in enumerate(track_data["bars"]):
            x = bar_index * 32
            
            for note_data in bar_data["notes"]:
                y = max_note - note_data["note"]
                assert y >= 0 and y < height, f"Invalid y: {y}, note {note_data['note']} min_note: {min_note}, max_note: {max_note}, difference: {max_note - min_note}, height: {height}"
                for i in range(note_data["start"], note_data["end"]):
                    image.putpixel((x + i, y), color)

    # Resize the image. Use nearest neighbor for pixel art.
    factor = 4
    image = image.resize((width * factor, height * factor), Image.NEAREST)

    return image


def convert_notesequence_to_wave(note_sequence):

    if len(note_sequence.notes) == 0:
        return None

    try:
        synthesizer = note_seq.fluidsynth
        wave = synthesizer(note_sequence, sample_rate=44100)
        return wave
    except Exception as e:
        synthesizer = note_seq.synthesize
        wave = synthesizer(note_sequence)
        return wave
    

def convert_notesequence_to_midi(note_sequence, filename="output.mid"):

    if len(note_sequence.notes) == 0:
        return None

    # Returns the file content of the midi file.
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
        note_seq.sequence_proto_to_midi_file(note_sequence, filename)
        with open(filename, "rb") as file:
            content = file.read()
    return content



