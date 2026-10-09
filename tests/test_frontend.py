import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from okur.frontend import SYMBOLS, Frontend, encode
from okur.frontend.circumflex import restore


@pytest.fixture(scope="module")
def frontend() -> Frontend:
    return Frontend()


@pytest.mark.parametrize(("text", "expected"), [
    ("Bu konuyla alakalı olarak şirketin karı hala artıyor.", "bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor."),
    ("Bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor.", "bu konuyla alâkalı olarak şirketin kârı hâlâ artıyor."),
    ("Dün gece çok kar yağdı.", "dün gece çok kar yağdı."),
    ("Halam hala bizde kalıyor.", "halam hâlâ bizde kalıyor."),
    ("Hala kızı geldi.", "hala kızı geldi."),
    ("Net kar yüzde 12 arttı.", "net kâr yüzde on iki arttı."),
    ("Dükkanlar kapandı, kağıdı aldım.", "dükkânlar kapandı, kâğıdı aldım."),
    ("Mekanik bir arıza var, mekanı değiştirdik.", "mekanik bir arıza var, mekânı değiştirdik."),
    ("Ofisimiz 3. katta.", "ofisimiz üçüncü katta."),
    ("1. Dünya Savaşı ile 2. 3. ve 4. sıralar", "birinci dünya savaşı ile ikinci üçüncü ve dördüncü sıralar"),
    ("Bütçe 1.250.000 TL.", "bütçe bir milyon iki yüz elli bin türk lirası."),
    ("İSTANBUL ılık", "i se te a ne be u le ılık"),  # capitals are spelled out by normalizer-tr
])
def test_frontend(frontend: Frontend, text: str, expected: str) -> None:
    assert frontend(text) == expected


def test_typed_circumflex_is_kept() -> None:
    assert restore("kâr ve kar") == "kâr ve kar"


@settings(max_examples=300, deadline=None)
@given(st.text(max_size=200))
def test_never_raises_and_stays_in_alphabet(text: str) -> None:
    out = Frontend()(text)
    assert set(out) <= set(SYMBOLS)
    assert all(i >= 2 for i in encode(out))  # never <pad> or <unk>
