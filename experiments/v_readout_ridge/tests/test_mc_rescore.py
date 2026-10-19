import pytest
from rescore_mc import parse_v2

@pytest.mark.parametrize('text,expected',[
    (r'Final Answer: \boxed{\text{D}}','D'),
    (r'Answer is A. Final: \boxed{\mathrm{C}}','C'),
    (r'\boxed{\mathbf{(B)}}','B'),
    ('**F**. an overarching explanation.','F'),
    ('**I.** Differential reproduction.','I'),
    ('The **correct answer** is **(C)**.','C'),
    ('A plausible explanation has several parts.',''),
    ('(A) first possibility\n(B) second possibility',''),
    ('<think>Answer is A.</think>\n**B**. final response','B'),
    ('<think>Answer is A.',''),
])
def test_format_normalization_without_reference(text,expected):
    assert parse_v2(text)==expected

def test_options_outside_question_range_remain_unparsed():
    assert parse_v2(r'\boxed{\text{I}}',4)==''
