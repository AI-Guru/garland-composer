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

 
import streamlit as st
from streamlit.components.v1 import html
from source.languagemodel import LanguageModel
from source.utilities import (
    convert_tokens_to_songdata,
    convert_songdata_to_notesequence,
    convert_songdata_to_pianoroll,
    convert_notesequence_to_wave,
    convert_notesequence_to_midi
)

# Set the page layout to "wide"
st.set_page_config(layout="wide")

# Define the MIDI instruments.
midi_instruments = {
    "Harpsichord": 6,
    "Church Organ": 19,
    "Piano": 0,
}

# Load the model once and cache it.
@st.cache_resource
def load_model():
    #model_id = "TristanBehrens/bach-garland-mambaplus"
    #model_id = "TristanBehrens/bach-garland-pharia"
    #model_id = "TristanBehrens/bach-garland-phariaplus"
    model_id = "TristanBehrens/bach-garland-phariaplusplus"
    model = LanguageModel(model_id)
    return model
model = load_model()


# Initialize token_sequence in session state if it doesn't exist
if "token_sequence" not in st.session_state:
    st.session_state.token_sequence = "GARLAND_START"
    st.session_state.song_data = None
    st.session_state.piano_roll = None
    st.session_state.wave = None
    st.session_state.note_sequence = None
    st.session_state.midi_file_content = None
    st.session_state.temperature = 0.1
    st.session_state.bpm = 100
    st.session_state.instrument = "Piano"


# Define the main function.
def main():

    # Create two columns.
    columns = st.columns([0.7, 0.3])

    # Set up the Streamlit application
    column = columns.pop(0)
    with column:

        # Change the colors of the a-tag to (255, 75, 75).
        st.markdown("<style>a:link { color: #FF4B4B; } a:visited { color: #FF4B4B; }</style>", unsafe_allow_html=True)

        # Add a title.
        st.title("Garland Composer")
        linkedin_url = "https://www.linkedin.com/dr-tristan-behrens-734967a2/"
        x_url = "https://x.com/DrTBehrens"
        st.write(f"By Dr. Tristan Behrens. Find me on [LinkedIn]({linkedin_url}) and [X]({x_url}).")
        hf_url = "https://huggingface.co/TristanBehrens/bach-garland-mambaplus/"
        st.write(f"Model available on [Hugging Face]({hf_url}).")

    # Add a picture.
    column = columns.pop(0)
    with column:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.image("assets/garland.jpg", use_column_width=True)

    # Add a horizontal line.
    st.markdown("---")

    # Create two columns.
    columns = st.columns(3)

    # Add a slider to control the temperature.
    state_temperature = st.session_state.temperature
    with columns.pop(0):
        temperature = st.slider("Temperature", 0.0, 1.0, state_temperature)
    st.session_state.temperature = temperature

    # Add a slider to control the bpm.
    state_bpm = st.session_state.bpm
    with columns.pop(0):
        bpm = st.slider("BPM", 80, 120, state_bpm, 5)
    st.session_state.bpm = bpm

    # Dropdown for the instrument.
    state_instrument = st.session_state.instrument
    with columns.pop(0):
        instrument = st.selectbox("Instrument", list(midi_instruments.keys()), index=list(midi_instruments.keys()).index(state_instrument))
    st.session_state.instrument = instrument
    
    # Get the token sequence from the session state.
    token_sequence = st.session_state.token_sequence

    # Columns for the buttons.
    columns = st.columns(5)

    # Add a button to generate the next bar.
    column = columns.pop(0)
    with column:
        if st.button("Add a bar", use_container_width=True):
            token_sequence = extend_sequence(model, token_sequence, temperature)
            refresh(token_sequence, bpm, instrument)

    # Add a button to compose long.
    column = columns.pop(0)
    with column:
        if st.button("Auto compose", use_container_width=True):
            token_sequence = auto_compose(model, token_sequence, temperature)
            refresh(token_sequence, bpm, instrument)

    # Add a button to remove the last bar.
    column = columns.pop(0)
    with column:
        if st.button("Remove last", use_container_width=True):
            token_sequence = shortened_sequence(token_sequence)
            refresh(token_sequence, bpm, instrument)

    # Add a button to reset the sequence.
    column = columns.pop(0)
    if token_sequence != "GARLAND_START":
        with column:
            if st.button("Reset", use_container_width=True):
                with columns.pop(0):
                    token_sequence = "GARLAND_START"
                    refresh(token_sequence, bpm, instrument)

    # Provide a download button for the MIDI file.
    column = columns.pop(0)
    if "midi_file_content" in st.session_state and st.session_state.midi_file_content is not None:
        with column:
            midi_file_content = st.session_state.midi_file_content
            if st.download_button(
                label="Download MIDI",
                data=midi_file_content,
                file_name="music.mid",
                mime="audio/midi",
                use_container_width=True
            ):
                pass

    # Add a horizontal line.
    st.markdown("---")

    # Display the piano roll.
    if "piano_roll" in st.session_state and st.session_state.piano_roll is not None:
        st.image(st.session_state.piano_roll)

    # Display an audio player.
    if "wave" in st.session_state and st.session_state.wave is not None:
        st.audio(st.session_state.wave, format="audio/wav", sample_rate=44100, autoplay=True)

    # Add a horizontal line.
    st.markdown("---")

    # Set the text color to (255, 31, 75).
    if token_sequence.endswith("GARLAND_END"):
        st.write("The AI believes that the music is finished.")
    else:
        st.write("The AI believes that the music is not finished.")



def auto_compose(model, token_sequence, temperature):

    max_iterations = 100
    for _ in range(max_iterations):
        token_sequence = extend_sequence(model, token_sequence, temperature)
        if token_sequence.endswith("GARLAND_END"):
            break
    return token_sequence


def extend_sequence(model, token_sequence, temperature):

    # Replace the last GARLAND_END token with NEXT.
    if token_sequence.endswith("GARLAND_END"):
        token_sequence = token_sequence.replace("GARLAND_END", "NEXT")

    # The maximum length of the generated music.
    max_length = 16_384

    # When to stop the generation.
    end_tokens = ["NEXT", "GARLAND_END"]

    # Compose the music iterativelybar by bar.
    output_dict = model.generate(
        prompt=token_sequence,
        temperature=temperature,
        max_length=max_length,
        end_tokens=end_tokens,
        forbidden_tokens=["[PAD]", "[EOS]"],
        return_structured_output=True
    )
    output = output_dict["output"]
    return output


def shortened_sequence(token_sequence):

    # Find the position of the next to last NEXT token.
    next_tokens = token_sequence.split()
    next_positions = [i for i, x in enumerate(next_tokens) if x == "NEXT" or x == "GARLAND_END"]
    if len(next_positions) <= 1:
        token_sequence = "GARLAND_START"
    else:
        next_position = next_positions[-2]
        token_sequence = " ".join(next_tokens[:next_position + 1])
    return token_sequence


def refresh(token_sequence="GARLAND_START", bpm=120, instrument="Piano"):

    try:

        # Convert to song data.
        song_data = convert_tokens_to_songdata(token_sequence)
        song_data["bpm"] = bpm

        # Set the instrument.
        for track in song_data["tracks"]:
            track["instrument"] = midi_instruments[instrument]

        # Convert to piano roll.
        piano_roll = convert_songdata_to_pianoroll(song_data)

        # Convert to note sequence.
        note_sequence = convert_songdata_to_notesequence(song_data)

        # Play the note sequence.
        wave = convert_notesequence_to_wave(note_sequence)

        # Get the MIDI file content.
        midi_file_content = convert_notesequence_to_midi(note_sequence)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return

    # Update the session state.
    st.session_state.token_sequence = token_sequence
    st.session_state.song_data = song_data
    st.session_state.piano_roll = piano_roll
    st.session_state.note_sequence = note_sequence
    st.session_state.wave = wave
    st.session_state.midi_file_content = midi_file_content

    # Rerun the app.
    st.rerun()


if __name__ == "__main__":
    main()
