# Garland Composer
# Copyright (c) 2024 Dr. Tristan Behrens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
 
import copy
import note_seq
from PIL import Image
import tempfile
import os
import colorama
from omegaconf import DictConfig, OmegaConf
import torch
from typing import List, Tuple, Dict
from dacite import from_dict
from collections.abc import MutableMapping
import sys


# NOTE: Imported from helibrunna.
def display_logo():
    """
    Display the logo by printing it line by line with a cyberpunk color scheme.

    Raises:
        FileNotFoundError: If the logo file is missing.
    """

    # Get the path of this script and use it to find the logo.
    script_path = os.path.dirname(os.path.realpath(__file__))
    search_path = os.path.dirname(script_path)

    # Load the logo.
    logo_path = os.path.join(search_path, "assets", "asciilogo.txt")
    if not os.path.exists(logo_path):
        raise FileNotFoundError("The logo file is missing.")
    with open(logo_path, "r") as f:
        logo = f.read()

    # Print the logo line by line. Use colorama to colorize the output. Use a cyberpunk color scheme.
    for line_index, line in enumerate(logo.split("\n")):
        color = colorama.Fore.GREEN
        style = colorama.Style.BRIGHT if line_index % 2 == 0 else colorama.Style.NORMAL
        print(color + style + line)
    print(colorama.Style.RESET_ALL)


# NOTE: Imported from helibrunna.
def model_from_config(model_config: DictConfig, device:str) -> torch.nn.Module:
    """
    Create a model based on the provided model configuration.

    Args:
        model_config (DictConfig): The configuration for the model.

    Returns:
        The created model.

    Raises:
        ValueError: If the model type is unknown.
    """
    
    # Get the model type from the configuration.
    model_type = model_config.get("type", "xLSTMLMModel")
    
    # Create the xLSTMLMModel.
    if model_type == "xLSTMLMModel":
        print("Creating xLSTMLMModel...")
        from xlstm.xlstm_lm_model import xLSTMLMModel, xLSTMLMModelConfig
        
        # If there is no GPU, use the vanilla backend.
        if not torch.cuda.is_available():
            #model_config.backend = "vanilla"
            model_config.slstm_block.slstm.backend = "vanilla"
            model_config.mlstm_block.mlstm.backend = "vanilla"
        model_config_object = from_dict(xLSTMLMModelConfig, OmegaConf.to_container(model_config))
        
        # Create the model.
        model = xLSTMLMModel(model_config_object)
        model.reset_parameters()
    
    # Create the GPT2LMModel.
    elif model_type == "gpt2":
        print("Creating GPT2LMModel...")
        from .models.gpttwo import GPT2LMModel, GPT2LMModelConfig
        model_config_object = from_dict(GPT2LMModelConfig, OmegaConf.to_container(model_config))
        model = GPT2LMModel(model_config_object)
    
    # Create the MambaLM.
    elif model_type == "mamba":
        print("Creating Mamba LM...")
        from mambapy.lm import LM, MambaConfig
        model_config_object = from_dict(MambaConfig, OmegaConf.to_container(model_config))
        model = LM(model_config_object, model_config.vocab_size)
    
    # Create the Transformer.
    elif model_type == "transformer":
        from .models.transformer import TransformerConfig, Transformer
        model_config_object = from_dict(TransformerConfig, OmegaConf.to_container(model_config))
        model = Transformer(model_config_object)
    
    # Create a Pharia instance.
    elif model_type == "pharia":
        from .models.pharia import PhariaConfig, PhariaModel
        model_config_object = from_dict(PhariaConfig, OmegaConf.to_container(model_config))
        model = PhariaModel(model_config_object)
    
    # Create a TransformerXL instance.
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move the model to the device.
    model.to(device)
    return model


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
    base_color = (255, 75, 75)
    adjustments = [1.2, 1.0, 0.8, 0.6]
    colors = []
    for adjustment in adjustments:
        import colorsys
        rgb = base_color
        rgb = [float(c) / 255.0 for c in rgb]
        hsv = colorsys.rgb_to_hsv(*rgb)
        offset = (adjustment - 1.0) * 0.1
        hsv = (hsv[0] + offset, hsv[1], hsv[2])
        rgb = colorsys.hsv_to_rgb(*hsv)
        rgb = tuple([int(255.0 * c) for c in rgb])
        colors += [rgb]

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

    # Set all the programs and instruments to 0.
    for note in note_sequence.notes:
        note.program = 0
        note.instrument = 0

    # Returns the file content of the midi file.
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        filename = temp_file.name
        note_seq.sequence_proto_to_midi_file(note_sequence, filename)
        with open(filename, "rb") as file:
            content = file.read()
    return content



