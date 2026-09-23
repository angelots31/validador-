"""
Pruebas del tokenizador de CLIP en Python puro (`app/ml/clip_tokenizer.py`).

El docstring del módulo afirmaba que sus ids "se validan en los tests contra
ejemplos conocidos de CLIP", pero esos tests no existían. Aquí están: se
comprueban contra el ejemplo canónico de `CLIPTokenizer`
(`"a photo of a cat"` -> `[49406, 320, 1125, 539, 320, 2368, 49407]`) y contra
el contrato de la API (`encode`), que es lo que consumiría una etapa de CLIP.

Este módulo sí es correcto y se conserva: el problema del commit 8 no fue el
tokenizador, sino los modelos ONNX, que están cuantizados con
`quantize_dynamic` y son inservibles (ver `scripts/export_clip_onnx.py`).
"""

import unittest
from pathlib import Path

from app.ml.clip_tokenizer import CONTEXT_LENGTH, ClipTokenizer, load_tokenizer

TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "ml" / "clip" / "tokenizer.json"
)

#: Ejemplo canónico de CLIPTokenizer (la secuencia completa, con BOS y EOS).
#: Si el BPE, la normalización o el mapeo byte->unicode se rompen, esto falla.
BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
CANONICAL_IDS = [BOS_TOKEN_ID, 320, 1125, 539, 320, 2368, EOS_TOKEN_ID]
#: Lo mismo sin los tokens especiales, que es lo que devuelve `encode_ids`.
CANONICAL_BPE_IDS = CANONICAL_IDS[1:-1]


@unittest.skipUnless(TOKENIZER_PATH.exists(), "falta app/ml/clip/tokenizer.json")
class ClipTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_tokenizer(TOKENIZER_PATH)

    def test_reproduces_the_canonical_clip_example(self):
        """La secuencia completa: BOS + BPE + EOS (y relleno hasta 77 aparte)."""
        ids = self.tokenizer.encode("a photo of a cat")
        self.assertEqual(ids[: len(CANONICAL_IDS)], CANONICAL_IDS)

    def test_encode_ids_leaves_out_the_special_tokens(self):
        self.assertEqual(
            self.tokenizer.encode_ids("a photo of a cat"), CANONICAL_BPE_IDS
        )

    def test_is_case_insensitive(self):
        """CLIP normaliza a minúsculas antes de tokenizar."""
        self.assertEqual(
            self.tokenizer.encode_ids("A PHOTO OF A CAT"), CANONICAL_BPE_IDS
        )

    def test_collapses_extra_whitespace(self):
        """El pre-tokenizado colapsa espacios: el resultado debe ser idéntico."""
        self.assertEqual(
            self.tokenizer.encode_ids("a photo   of  a cat"), CANONICAL_BPE_IDS
        )

    def test_encode_pads_to_the_context_length(self):
        """CLIP fue entrenado con secuencias de 77 posiciones."""
        self.assertEqual(len(self.tokenizer.encode("a photo of a cat")), CONTEXT_LENGTH)
        self.assertEqual(len(self.tokenizer.encode("papaya")), CONTEXT_LENGTH)

    def test_encode_truncates_very_long_text(self):
        ids = self.tokenizer.encode("fruta " * 300)
        self.assertEqual(len(ids), CONTEXT_LENGTH)
        # Aun truncado, debe terminar en EOS: es la posición de la que CLIP lee.
        self.assertEqual(ids[-1], EOS_TOKEN_ID)

    def test_different_words_produce_different_tokens(self):
        """Un tokenizador que devolviera lo mismo para todo sería inútil."""
        papaya = self.tokenizer.encode_ids("papaya")
        carrot = self.tokenizer.encode_ids("carrot")
        self.assertNotEqual(papaya, carrot)

    def test_encodes_non_ascii_text_without_crashing(self):
        """Los prompts pueden llevar tildes: la normalización NFC debe cubrirlo."""
        ids = self.tokenizer.encode_ids("una foto de un melón")
        self.assertTrue(ids)
        self.assertTrue(all(isinstance(token_id, int) for token_id in ids))

    def test_encode_batch_matches_encode(self):
        texts = ["a photo of a papaya", "a photo of a carrot"]
        self.assertEqual(
            self.tokenizer.encode_batch(texts),
            [self.tokenizer.encode(text) for text in texts],
        )

    def test_load_tokenizer_is_cached(self):
        """Cargar el vocabulario dos veces debe devolver el mismo objeto."""
        self.assertIs(load_tokenizer(TOKENIZER_PATH), load_tokenizer(TOKENIZER_PATH))

    def test_missing_tokenizer_file_raises_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            ClipTokenizer(TOKENIZER_PATH.parent / "no-existe.json")


if __name__ == "__main__":
    unittest.main()
