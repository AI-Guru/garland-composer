

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

modes = ["major", "minor"]

modes_to_intervals = {
    "major": [2, 2, 1, 2, 2, 2, 1],
    "minor": [2, 1, 2, 2, 1, 2, 2]
}

def get_inverted_pitches(root_note, mode):
    pitches = get_pitches(root_note, mode)
    all_pitches = list(range(128))
    inverted_pitches = [pitch for pitch in all_pitches if pitch not in pitches]
    return inverted_pitches

def get_pitches(root_note, mode):
    assert root_note in notes, "Invalid root note"
    assert mode in modes, "Invalid mode"
    pitches = []
    pitch = notes.index(root_note)
    for interval in modes_to_intervals[mode][:]:
        pitches.append(pitch)
        for octave in range(1, 11):
            if pitch + octave * 12 < 128:
                pitches.append(pitch + octave * 12)
        pitch = pitch + interval
    assert len(set(pitches)) == len(pitches), "Duplicate pitches"
    pitches = sorted(pitches)
    return pitches