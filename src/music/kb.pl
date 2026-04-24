% kb.pl  --  music knowledge base for the piano-playing arm
%
% Notes, intervals, scales, and simple melodies.
% All note names match the Python ALL_KEYS dict (e.g. 'C4', 'C#4').

% --- MIDI pitch numbers ---
midi('C4',  60). midi('C#4', 61). midi('D4',  62). midi('D#4', 63).
midi('E4',  64). midi('F4',  65). midi('F#4', 66). midi('G4',  67).
midi('G#4', 68). midi('A4',  69). midi('A#4', 70). midi('B4',  71).
midi('C5',  72).

% semitone_distance(+N1, +N2, -D)
semitone_distance(N1, N2, D) :-
    midi(N1, M1), midi(N2, M2),
    D is abs(M2 - M1).

% step/2  --  two notes are a step apart (<=2 semitones)
step(N1, N2) :- semitone_distance(N1, N2, D), D =< 2.

% leap/2  --  two notes are a leap (>2 semitones)
leap(N1, N2) :- semitone_distance(N1, N2, D), D > 2.

% --- C-major scale ---
c_major('C4'). c_major('D4'). c_major('E4'). c_major('F4').
c_major('G4'). c_major('A4'). c_major('B4'). c_major('C5').

in_scale(Note, c_major) :- c_major(Note).

% --- Built-in melodies ---
% melody(+Name, -NoteList)
melody(twinkle,
    ['C4','C4','G4','G4','A4','A4','G4',
     'F4','F4','E4','E4','D4','D4','C4']).

melody(mary,
    ['E4','D4','C4','D4','E4','E4','E4',
     'D4','D4','D4','E4','G4','G4']).

melody(ode_to_joy,
    ['E4','E4','F4','G4','G4','F4','E4','D4',
     'C4','C4','D4','E4','E4','D4','D4']).

% all_in_scale(+Notes, +Scale) -- true if every note is in the scale
all_in_scale([], _).
all_in_scale([H|T], Scale) :- in_scale(H, Scale), all_in_scale(T, Scale).
