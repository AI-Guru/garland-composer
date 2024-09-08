import streamlit as st
import sys
sys.path.append("../helibrunna")
from source.onnxlanguagemodel import OnnxLanguageModel
from PIL import Image
import random

# Load the model once and cache it.
@st.cache_resource
def load_model():
    return OnnxLanguageModel("TristanBehrens/bach-garland-mambaplus")
model = load_model()

# Initialize token_sequence in session state if it doesn't exist
if "token_sequence" not in st.session_state:
    st.session_state.token_sequence = "GARLAND_START"

# Add a random image to the state if it doesn't exist.
if "piano_roll" not in st.session_state:
    width = 128
    height = 128
    image = Image.new("RGB", (width, height))
    # Add random pixels to the image.
    for x in range(width):
        for y in range(height):
            image.putpixel((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))            
    st.session_state.piano_roll = image

def main():
    # Set up the Streamlit application
    st.title("Simple Streamlit Application")

    # Display a picture
    st.image(st.session_state.piano_roll, caption="Sample Image")
    
    token_sequence = st.session_state.token_sequence

    # Display the token sequence.
    st.write("Token sequence:", token_sequence)

    # Add two buttons: Yes and No
    if st.button("Yes"):
        token_sequence = extend_sequence(model, token_sequence)
        st.session_state.token_sequence = token_sequence
        refresh()

    if st.button("No"):
        token_sequence = shortened_sequence(token_sequence)
        st.session_state.token_sequence = token_sequence
        refresh()

def extend_sequence(model, token_sequence):
    # The temperature of the generation. The higher the temperature, the more random the output.
    temperature = 0.5

    # The maximum length of the generated music.
    max_length = 16_384

    # When to stop the generation.
    end_tokens = ["NEXT"]

    # Compose the music iterativelybar by bar.
    output_dict = model.generate(
        prompt=token_sequence,
        temperature=temperature,
        max_length=max_length,
        end_tokens=end_tokens,
        forbidden_tokens=["[PAD]", "[EOS]", "GARLAND_END"],
        return_structured_output=True
    )
    output = output_dict["output"]
    return output

def shortened_sequence(token_sequence):

    # Find the position of the next to last NEXT token.
    next_tokens = token_sequence.split()
    next_positions = [i for i, x in enumerate(next_tokens) if x == "NEXT"]
    if len(next_positions) <= 1:
        token_sequence = "GARLAND_START"
    else:
        next_position = next_positions[-2]
        token_sequence = " ".join(next_tokens[:next_position + 1])
    return token_sequence

def refresh():

    # 

    st.rerun()

if __name__ == "__main__":
    main()