#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (c) 2014-2018
Gonzalo Ciruelos <gonzalo.ciruelos@gmail.com>
Federico Ferri <federico.ferri.it@gmail.com>
"""

import re
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
)

T = TypeVar("T")


def UnsupportedOperands(op: str, type1: object, type2: object) -> Exception:
    """Raise Type Error for unsupported operands."""

    def as_type(x: Union[T, Type[T]]) -> Type[T]:
        return x if isinstance(x, type) else type(x)

    fmt = "unsupported operand type(s) for {}: '{}' and '{}'"
    type1 = as_type(type1)
    type2 = as_type(type2)
    return TypeError(fmt.format(op, type1.__name__, type2.__name__))


class Letter:
    """
    The letter class.

    There are 7 letters: C, D, E, F, G, A, and B.

    This class implements basic letter arithmetic, such as adding and
    subtracting interval numbers, or computing a difference between
    two letters.
    """

    letters: str = "CDEFGAB"
    letters_idx: Dict[str, int] = {x: i for i, x in enumerate(letters)}
    letters_number: Dict[str, int] = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    @staticmethod
    def all() -> Iterator["Letter"]:
        """Return iterator for all letters."""
        for name in Letter.letters:
            yield Letter(name)

    def __init__(self, letter: str) -> None:
        """Initialize Letter class."""
        if letter not in self.letters_idx:
            raise ValueError("Invalid letter {!r}".format(letter))
        self.name: str = letter
        self.idx: int = self.letters_idx[letter]

    def __add__(self, other: int) -> "Letter":
        """Get the note letter by adding an integer to the current letter."""
        if isinstance(other, int):
            if other == 0:
                raise ValueError("Invalid interval number: 0")
            new_idx = (self.idx + other - (1 if other > 0 else -1)) % len(
                self.letters
            )
            return Letter(self.letters[new_idx])
        else:
            raise UnsupportedOperands("+", self, other)

    @overload
    def __sub__(self, other: "Letter") -> int:
        """Overload for dunder sub with letter as input."""

    @overload
    def __sub__(self, other: int) -> "Letter":
        """Overload for dunder sub with int as input."""

    def __sub__(self, other: Union[int, "Letter"]) -> Union[int, "Letter"]:
        """Subtract either a note Letter or an integer from current Letter."""
        if isinstance(other, Letter):
            d = self.idx - other.idx
            d += 1 if d >= 0 else -1
            return d
        elif isinstance(other, int):
            return self + -other
        else:
            raise UnsupportedOperands("-", self, other)

    def __str__(self) -> str:
        """Return string version of Letter."""
        return self.name

    def __repr__(self) -> str:
        """Return the repr format of this class."""
        return "Letter({!r})".format(str(self))

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        return str(self) == str(other)

    def number(self) -> int:
        """Return letter as number indicating the scale position."""
        return self.letters_number[self.name]

    def has_flat(self) -> bool:
        """Check if this note Letter can have flat notes."""
        return self.name not in "CF"

    def has_sharp(self) -> bool:
        """Check if this note letter can have sharp values."""
        return self.name not in "EB"


class Note:
    """
    The note class.

    The notes are to be parsed in the following way:
    * the letter name,
    * accidentals (up to 3),
    * octave (default is 4).

    For example, 'Ab', 'G9', 'B##7' are all valid notes. '#', 'A9b',
    'Dbbbb' are not.
    """

    pattern = re.compile(r"([A-G])(b{0,3}|#{0,3})(\d{0,1})$")

    @staticmethod
    def all(min_octave: int = 4, max_octave: int = 4) -> Iterator["Note"]:
        """Return an iterator containing all notes within the given octaves."""
        for octave in range(min_octave, max_octave + 1):
            for letter in Letter.all():
                letter_accidentals = [""]
                if letter.has_flat():
                    letter_accidentals.insert(0, "b")
                if letter.has_sharp():
                    letter_accidentals.append("#")
                for acc in letter_accidentals:
                    yield Note("{}{}{:d}".format(letter.name, acc, octave))

    @staticmethod
    def accidental_value(acc: str) -> int:
        """Return the modifier of the given accidental."""
        if acc == "":
            return 0
        return {"#": 1, "b": -1}[acc[0]] + Note.accidental_value(acc[1:])

    @staticmethod
    def accidental_str(val: int) -> str:
        """Return the string part of an accidental based on offset."""
        return "b" * max(0, -val) + "#" * max(0, val)

    def __init__(self, note: str) -> None:
        """Create Note class."""
        m = self.pattern.match(note)
        if m is None:
            raise ValueError("Could not parse the note {!r}".format(note))

        self.letter = Letter(m.group(1))
        self.accidental = m.group(2)
        self.octave = int(m.group(3) or "4")

        self.number = (
            self.letter.number()
            + self.octave * 12
            + Note.accidental_value(self.accidental)
        )

    def __add__(self, other: object) -> "Note":
        """Get a note that is offset by a given interval."""
        if isinstance(other, Interval):
            if other.is_compound():
                from functools import reduce

                return reduce(lambda a, b: a + b, other.split(), self)

            new_letter = self.letter + other.number
            new_number = self.number + other.semitones
            new_note_octave = self.octave + int(
                self.letter.name in Letter.letters[8 - other.number :]
            )
            difference = new_number % 12 - new_letter.number()
            if difference < -3:
                difference += 12
            if difference > 3:
                difference -= 12
            return Note(
                new_letter.name
                + Note.accidental_str(difference)
                + str(new_note_octave)
            )
        else:
            raise UnsupportedOperands("+", self, other)

    @overload
    def __sub__(self, other: "Note") -> "Interval":
        """Overload for dunder sub with Note as input."""

    @overload
    def __sub__(self, other: "Interval") -> "Note":
        """Overload for dunder sub with Interval as input."""

    def __sub__(
        self, other: Union["Note", "Interval"]
    ) -> Union["Note", "Interval"]:
        """Find a lower note or an interval between two notes."""
        if isinstance(other, Interval):
            if other.is_compound():
                from functools import reduce

                return reduce(lambda a, b: a - b, other.split(), self)

            return self.to_octave(self.octave - 1) + other.complement()
        elif isinstance(other, Note):
            notes = list((n.midi_note(), n) for n in (self, other))
            semitones = notes[0][0] - notes[1][0]
            if semitones < -1:
                raise ArithmeticError("Interval smaller than d1")
            number = notes[0][1].letter - notes[1][1].letter
            octaves = 0
            while semitones >= 12:
                semitones -= 12
                octaves += 1
            number = (number + (1 if number < 0 else -1)) % 7 + 1
            for i in Interval.all():
                if i.number == number and i.semitones == semitones:
                    return Interval(i.quality + str(octaves * 7 + number))
            raise ValueError("Interval N={} S={}".format(number, semitones))
        else:
            raise UnsupportedOperands("-", self, other)

    def midi_note(self) -> int:
        """Return the midi note value."""
        return self.number + 12

    def frequency(self) -> float:
        """Return the harmonic frequency of a note."""
        from math import pow

        return 440.0 * pow(2, (self.number - Note("A4").number) / 12.0)

    def to_octave(self, octave: int) -> "Note":
        """Get a note in a different octave."""
        return Note(self.letter.name + self.accidental + str(octave))

    @property
    def enharmonic_equivalent(self) -> "Note":
        """Return the simple enharmonic equivalent variant.

        This translates E# to F, Cb to B, etc.
        """
        if self.accidental == "#" and self.letter.has_sharp() is False:
            octave = self.octave
            if str(self.letter) == "B":
                octave += 1

            return Note(f"{self.letter - 2}{octave}")
        if self.accidental == "b" and self.letter.has_flat():
            octave = self.octave
            if str(self.letter) == "C":
                octave += -1

            return Note(f"{self.letter + 2}{octave}")

    def lilypond_notation(self) -> str:
        """Get the note in lilypond notation."""
        return str(self).replace("b", "es").replace("#", "is").lower()

    def scientific_notation(self) -> str:
        """Get the note in scientific notation."""
        return str(self) + str(self.octave)

    def __repr__(self) -> str:
        """Get the class repr format."""
        return "Note({!r})".format(self.scientific_notation())

    def __str__(self) -> str:
        """Get the note as a string."""
        return self.letter.name + self.accidental

    def __eq__(self, other: object) -> bool:
        """Compare note with other note."""
        if isinstance(other, Note):
            return self.scientific_notation() == other.scientific_notation()

        raise UnsupportedOperands("==", self, other)


class Interval:
    """
    The interval class.

    The intervals are to be parsed in th following way:
    * the quality, (m, M, p, A, d)
    * the number.

    For example, 'd8', 'P1', 'A5' are valid intervals. 'P3', '5' are not.
    """

    intervals: Dict[str, int] = {
        "d1": -1,
        "P1": 0,
        "A1": 1,
        "d2": 0,
        "m2": 1,
        "M2": 2,
        "A2": 3,
        "d3": 2,
        "m3": 3,
        "M3": 4,
        "A3": 5,
        "d4": 4,
        "P4": 5,
        "A4": 6,
        "d5": 6,
        "P5": 7,
        "A5": 8,
        "d6": 7,
        "m6": 8,
        "M6": 9,
        "A6": 10,
        "d7": 9,
        "m7": 10,
        "M7": 11,
        "A7": 12,
        "d8": 11,
        "P8": 12,
        "A8": 13,
    }
    quality_inverse: Dict[str, str] = {
        "P": "P",
        "d": "A",
        "A": "d",
        "m": "M",
        "M": "m",
    }

    @staticmethod
    def all() -> Iterator["Interval"]:
        """Create an iterator for all defined intervals."""
        for name in Interval.intervals:
            yield Interval(name)

    def __init__(self, interval: str) -> None:
        """Create Interval class."""
        self.quality: str = interval[0]
        self.number: int = int(interval[1:])
        self.semitones: int = 0

        # compound intervals:
        number = self.number
        while number > 8:
            number -= 7
            self.semitones += 12
        interval1 = self.quality + str(number)

        try:
            self.semitones += self.intervals[interval1]
        except KeyError:
            raise ValueError("Invalid interval {!r}.".format(interval))

    def __str__(self) -> str:
        """Return interval as a string."""
        return self.quality + str(self.number)

    def __repr__(self) -> str:
        """Return class repr format."""
        return "Interval({!r})".format(str(self))

    def __eq__(self, other: object) -> bool:
        """Compare interval with other interval."""
        if isinstance(other, Interval):
            return str(self) == str(other)
        if isinstance(other, str):
            return str(self) == other
        raise UnsupportedOperands("==", self, other)

    def is_compound(self) -> bool:
        """Check if interval is a compound interval."""
        return self.number > 8

    def split(self) -> List["Interval"]:
        """Split a compound interval into simple intervals.

        The sum of splitted intervals is equal to the compound interval.
        """
        ret: List[Interval] = []
        i = Interval(str(self))
        while i.is_compound():
            i.number -= 7
            i.semitones -= 12
            ret.append(Interval("P8"))
        ret.append(i)
        return ret

    def complement(self) -> "Interval":
        """
        Return the complement of this interval also known as inverted interval.

        The sum of this interval plus its complement is equal to 1 octave (P8),
        except for the case of A8, for which there is no d1 interval.
        """
        if self.is_compound():
            raise ValueError("Cannot invert a compound interval")
        else:
            n = 9 - self.number
            q = self.quality_inverse[self.quality]
            return Interval(q + str(n))


RootType = Union[Sequence[Note], Note]


class Chord:
    """
    The chord class.

    Contains recipes for common chords.
    """

    recipes: Dict[str, List[str]] = {
        "maj": ["P1", "M3", "P5"],
        "min": ["P1", "m3", "P5"],
        "aug": ["P1", "M3", "A5"],
        "dim": ["P1", "m3", "d5"],
        "dom7": ["P1", "M3", "P5", "m7"],
        "min7": ["P1", "m3", "P5", "m7"],
        "maj7": ["P1", "M3", "P5", "M7"],
        "aug7": ["P1", "M3", "A5", "m7"],
        "dim7": ["P1", "m3", "d5", "d7"],
        "m7dim5": ["P1", "m3", "d5", "m7"],
        "sus2": ["P1", "P5", "P8", "M2"],
        "sus4": ["P1", "P5", "P8", "P4"],
        "open5": ["P1", "P5", "P8"],
        "dom9": ["P1", "M3", "P5", "m7", "M9"],
        "min9": ["P1", "m3", "P5", "m7", "M9"],
        "maj9": ["P1", "M3", "P5", "M7", "M9"],
        "aug9": ["P1", "M3", "A5", "m7", "M9"],
        "dim9": ["P1", "m3", "d5", "d7", "M9"],
    }
    aliases: Dict[str, str] = {
        "M": "maj",
        "m": "min",
        "+": "aug",
        "°": "dim",
        "7": "dom7",
        "m7": "min7",
        "M7": "maj7",
        "+7": "aug7",
        "7aug5": "aug7",
        "7#5": "aug7",
        "°7": "m7dim5",
        "ø7": "m7dim5",
        "m7b5": "m7dim5",
        "9": "dom9",
        "m9": "min9",
        "M9": "maj9",
        "+9": "aug9",
        "°9": "dim9",
    }

    adds: Dict[str, str] = {
        "4": "P4",
        "9": "M9",
        "11": "M2",
    }

    valid_types: List[str] = list(recipes.keys()) + list(aliases.keys())

    @staticmethod
    def all(
        min_octave: int = 4,
        max_octave: int = 4,
        root: Optional[RootType] = None,
    ) -> Iterator["Chord"]:
        """Return an iterator over all known chord types."""
        roots: Union[Iterator[Note], Sequence[Note]]
        if root is None:
            roots = Note.all()
        elif isinstance(root, (list, set, tuple)):
            roots = root
        elif isinstance(root, Note):
            roots = [root]
        else:
            raise TypeError("Invalid root type: {}".format(type(root)))
        for root_note in roots:
            for name in Chord.recipes:
                yield Chord(root_note, name)

    def __init__(self, root: Union[str, Note], chord_type: str = "M") -> None:
        """Create Chord class."""
        add: Optional[str] = None

        if isinstance(root, str):
            if "add" in root:
                root, add = root.split("add")
            for s in sorted(self.valid_types, key=lambda x: -len(x)):
                if root.endswith(s):
                    chord_type = s
                    root = Note(root[: -len(s)])
                    break
            if not isinstance(root, Note):
                raise ValueError("Invalid chord: {!r}".format(root))

        if "add" in chord_type:
            chord_type, add = chord_type.split("add")

        if chord_type in self.aliases:
            chord_type = self.aliases[chord_type]
        if chord_type not in self.recipes.keys():
            raise ValueError("Invalid chord type: {}.".format(chord_type))

        self.chord_type = chord_type
        self.root = root

        self.add = add

    @property
    def notes(self) -> List[Note]:
        """Return notes in the chord."""
        notes = [
            self.root + Interval(i) for i in self.recipes[self.chord_type]
        ]

        if self.add is None:
            return notes

        add = self.add
        chord_type = self.chord_type
        root = self.root
        if add not in self.adds:
            raise ValueError(f"Unknown chord type: {chord_type}add{add}")
        add_interval = self.adds[add]
        extra_note = root + Interval(add_interval)

        notes.append(extra_note)
        return sorted(notes, key=lambda x: x.frequency())

    @property
    def simplified_notes(self) -> List[Note]:
        """Return notes in the chord.

        All enharmonic variants of notes will use the simplest form
        e.g., E# will be return as F.
        """
        return [note.enharmonic_equivalent for note in self.notes]

    def __repr__(self) -> str:
        """Return chord class repr."""
        return "Chord({!r}, {!r})".format(self.notes[0], self.chord_type)

    def __str__(self) -> str:
        """Return chord as string."""
        return "{}{}".format(str(self.notes[0]), self.chord_type)

    def __eq__(self, other: object) -> bool:
        """Compare chord instance with another chord."""
        if not isinstance(other, Chord):
            raise UnsupportedOperands("==", self, other)

        if len(self.notes) != len(other.notes):
            # if chords dont have the same number of notes, def not equal
            return False
        else:
            return all(
                self.notes[i] == other.notes[i] for i in range(len(self.notes))
            )


class Scale:
    """
    The scale class.

    Contains recipes for common scales, and operators for accessing the
    notes of the scale, and for checking if a scale contains specific
    notes or chords.
    """

    scales: Dict[str, List[str]] = {
        "major": ["P1", "M2", "M3", "P4", "P5", "M6", "M7"],
        "natural_minor": ["P1", "M2", "m3", "P4", "P5", "m6", "m7"],
        "harmonic_minor": ["P1", "M2", "m3", "P4", "P5", "m6", "M7"],
        "melodic_minor": ["P1", "M2", "m3", "P4", "P5", "M6", "M7"],
        "major_pentatonic": ["P1", "M2", "M3", "P5", "M6"],
        "minor_pentatonic": ["P1", "m3", "P4", "P5", "m7"],
        # greek modes:
        "ionian": ["P1", "M2", "M3", "P4", "P5", "M6", "M7"],
        "dorian": ["P1", "M2", "m3", "P4", "P5", "M6", "m7"],
        "phrygian": ["P1", "m2", "m3", "P4", "P5", "m6", "m7"],
        "lydian": ["P1", "M2", "M3", "A4", "P5", "M6", "M7"],
        "mixolydian": ["P1", "M2", "M3", "P4", "P5", "M6", "m7"],
        "aeolian": ["P1", "M2", "m3", "P4", "P5", "m6", "m7"],
        "locrian": ["P1", "m2", "m3", "P4", "d5", "m6", "m7"],
    }
    greek_modes: Dict[int, str] = {
        1: "ionian",
        2: "dorian",
        3: "phrygian",
        4: "lydian",
        5: "mixolydian",
        6: "aeolian",
        7: "locrian",
    }
    greek_modes_set = set(greek_modes.values())

    @staticmethod
    def all(include_greek_modes: bool = False) -> Iterator["Scale"]:
        """Create an iterator that returns all scales."""
        for root in Note.all():
            for name in Scale.scales:
                if not include_greek_modes and name in Scale.greek_modes_set:
                    continue
                yield Scale(root, name)

    def __init__(self, root: Note, name: str) -> None:
        """Create Scale class instance."""
        if isinstance(root, str):
            root = Note(root)

        if not isinstance(root, Note):
            raise TypeError("Invalid root note type: {}".format(type(root)))
        if name not in self.scales:
            raise NameError("No such scale: {}".format(name))

        self.root = root
        self.name = name
        self.intervals = [Interval(i) for i in self.scales[name]]
        self.notes = [(root + i).to_octave(0) for i in self.intervals]
        self.max_index: int = len(self.notes) - 1

    @overload
    def __getitem__(self, k: int) -> Note:
        """Overload for dunder getitem with integer as input."""

    @overload
    def __getitem__(self, k: slice) -> List[Note]:
        """Overload for dunder getitem with slice as input."""

    def __getitem__(self, k: Union[int, slice]) -> Union[Note, List[Note]]:
        """Get an element from the scale at a given index."""
        if isinstance(k, int):
            try:
                octaves = k // len(self)
                offset = k - octaves * len(self)
                return (
                    self.root.to_octave(self.root.octave + octaves)
                    + self.intervals[offset]
                )
            except ValueError:
                raise IndexError("Index out of range")
        elif isinstance(k, slice):
            start = k.start or 0
            stop = k.stop
            step = k.step or 1
            return [self[i] for i in range(start, stop, step)]
        else:
            raise TypeError("Scale cannot be indexed by {}.".format(type(k)))

    def __len__(self) -> int:
        """Return the amount of notes in the Scale."""
        return len(self.intervals)

    def __contains__(self, k: object) -> bool:
        """Check if a scale contains a certain note or notes."""
        if isinstance(k, Note):
            return k.to_octave(0) in self.notes
        elif isinstance(k, Chord):
            return all(n in self for n in k.notes)
        elif isinstance(k, (list, set, tuple)):
            return all(x in self for x in k)
        else:
            return False

    def __str__(self) -> str:
        """Return scale as string."""
        return "{} {}".format(self.root, self.name)

    def __repr__(self) -> str:
        """Return class repr format."""
        return "Scale({!r}, {!r})".format(self.root, self.name)
