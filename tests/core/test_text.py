"""Text from bytes, with the error a person can act on: the file, the line and the encoding."""

import codecs

import pytest

from gelpy.core.errors import InvalidValueError, to_error_response
from gelpy.core.text import decode_utf8, line_of


@pytest.mark.parametrize(
    ("data", "offset", "line"),
    [
        (b"abc", 0, 1),
        (b"a\nb\nc", 4, 3),
        (b"a\r\nb\r\nc", 6, 3),  # CR LF is one line ending
        (b"a\rb\rc", 4, 3),  # and so is a lone CR: the endings of an old Mac file
        (b"a\rb\nc\r\nd", 7, 4),  # in any mixture
        (b"a\n\n\nb", 4, 4),
        (b"", 0, 1),
    ],
)
def test_the_line_of_an_offset_counts_every_kind_of_line_ending(data, offset, line):
    assert line_of(data, offset) == line


def test_valid_text_is_returned_as_text():
    assert decode_utf8("Größe\n".encode("utf-8"), what="The file 'x'") == "Größe\n"


def test_a_byte_order_mark_is_dropped_only_when_asked():
    data = codecs.BOM_UTF8 + b"label"
    assert decode_utf8(data, what="The file 'x'", bom=True) == "label"
    assert decode_utf8(data, what="The file 'x'") == chr(0xFEFF) + "label"


@pytest.mark.parametrize("ending", ["\r", "\r\n", "\n"], ids=["CR", "CR LF", "LF"])
def test_a_byte_that_is_not_utf8_is_refused_by_name_with_its_line_and_the_encoding(ending):
    data = ending.encode().join([b"Probe", b"Messung", b"Gr\xf6\xdfe", b"Ende"])
    with pytest.raises(InvalidValueError) as info:
        decode_utf8(data, what="The labels file 'labels.txt'")
    response = to_error_response(info.value)
    assert response.code == "invalid_value"
    assert response.error == "The labels file 'labels.txt' must be UTF-8 text, but line 3 is not valid UTF-8."
    assert "UTF-8" in response.detail and "bug" not in response.detail


def test_the_next_step_can_be_the_callers_own():
    with pytest.raises(InvalidValueError) as info:
        decode_utf8(b"\xff", what="The file 'x'", detail="Save x as UTF-8 text, one label per line.")
    assert info.value.detail == "Save x as UTF-8 text, one label per line."


def test_the_line_after_a_byte_order_mark_counts_from_the_first_line():
    data = codecs.BOM_UTF8 + b"Probe\nGr\xf6\xdfe\n"
    with pytest.raises(InvalidValueError) as info:
        decode_utf8(data, what="The file 'x'", bom=True)
    assert "line 2" in str(info.value)
