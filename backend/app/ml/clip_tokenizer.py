"""Tokenizador CLIP en Python puro.

El modelo de texto de CLIP espera ``input_ids``; para obtenerlos normalmente se
depende de ``tokenizers``/``transformers``, que arrastran cientos de MB. Como el
tokenizador de CLIP es un BPE byte-level con vocabulario fijo, se puede
reimplementar aqui en ~150 lineas y ahorrar toda esa dependencia.

Se reproduce exactamente la configuracion de ``CLIPTokenizer``:

* normalizacion NFC, colapso de espacios y minusculas;
* pre-tokenizado con la expresion regular de CLIP;
* ByteLevel + BPE con sufijo de fin de palabra ``</w>``;
* envoltura ``<|startoftext|> ... <|endoftext|>`` y relleno hasta 77.

Los ids resultantes se validan en los tests contra ejemplos conocidos de CLIP
(por ejemplo ``"a photo of a cat"`` -> ``[49406, 320, 1125, 539, 320, 2368, 49407]``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

#: Tokens especiales del vocabulario CLIP.
BOS_TOKEN_ID = 49406
EOS_TOKEN_ID = 49407
#: CLIP fue entrenado con secuencias de 77 posiciones.
CONTEXT_LENGTH = 77

#: Sufijo que marca el final de palabra en el BPE de CLIP.
END_OF_WORD = "</w>"

#: Tokens especiales que se codifican literalmente, sin pasar por el BPE.
SPECIAL_TOKENS = {
    "<|startoftext|>": BOS_TOKEN_ID,
    "<|endoftext|>": EOS_TOKEN_ID,
}


def _bytes_to_unicode() -> dict[int, str]:
    """Mapeo byte -> caracter imprimible usado por el BPE byte-level de GPT-2.

    Es el mismo mapeo que usa CLIP: los bytes imprimibles se quedan igual y el
    resto se desplaza a partir del caracter 256 para que todo sea representable
    como ``str``.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    mapping = dict(zip(printable, printable))
    offset = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = 256 + offset
            offset += 1
    return {byte: chr(codepoint) for byte, codepoint in mapping.items()}


BYTE_ENCODER = _bytes_to_unicode()

#: Equivalente en `re` de la expresion regular oficial de CLIP:
#: ``'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+``
#: (`re` no soporta ``\p{L}``, de ahi ``[^\W\d_]`` para "solo letras").
_LETTER = r"[^\W\d_]"
_DIGIT = r"\d"
TOKEN_PATTERN = re.compile(
    r"<\|startoftext\|>|<\|endoftext\|>"
    r"|'s|'t|'re|'ve|'m|'ll|'d"
    rf"|{_LETTER}+"
    rf"|{_DIGIT}"
    rf"|[^\s\w]+|_",
    re.IGNORECASE,
)

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalizacion NFC, colapso de espacios en blanco y minusculas."""
    text = unicodedata.normalize("NFC", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text.lower()


class ClipTokenizer:
    """BPE byte-level de CLIP cargado desde un ``tokenizer.json``."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        with open(tokenizer_path, encoding="utf-8") as handle:
            payload = json.load(handle)

        model = payload["model"]
        self.vocab: dict[str, int] = model["vocab"]
        #: rank de cada par ("a b" -> indice) para aplicar los merge en orden.
        self.merges: dict[tuple[str, str], int] = {
            tuple(merge.split(" ")): rank  # type: ignore[misc]
            for rank, merge in enumerate(model["merges"])
        }

    def _bpe(self, token: str) -> list[str]:
        """Aplica los merge del BPE a un pre-token ya codificado en bytes.

        Se marca el ultimo simbolo con ``</w>`` para que el vocabulario sepa que
        ahi termina la palabra, igual que en el BPE original de CLIP.
        """
        if not token:
            return []
        word: tuple[str, ...] = tuple(token[:-1]) + (token[-1] + END_OF_WORD,)
        if len(word) == 1:
            return [word[0]]

        pairs = _get_pairs(word)
        while pairs:
            bigram = min(pairs, key=lambda pair: self.merges.get(pair, float("inf")))
            if bigram not in self.merges:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    jump = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:jump])
                index = jump
                if (
                    word[index] == first
                    and index < len(word) - 1
                    and word[index + 1] == second
                ):
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        return list(word)

    def encode_ids(self, text: str) -> list[int]:
        """Devuelve los ids sin tokens especiales ni relleno."""
        ids: list[int] = []
        for chunk in TOKEN_PATTERN.findall(normalize_text(text)):
            # Solo los tokens especiales se toman literales. Ojo: palabras como
            # "cat" tambien existen en el vocabulario, pero *sin* el sufijo de
            # fin de palabra, asi que nunca deben resolverse por esta via.
            if chunk in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[chunk])
                continue
            encoded = "".join(BYTE_ENCODER[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(encoded):
                token_id = self.vocab.get(piece)
                if token_id is not None:
                    ids.append(token_id)
        return ids

    def encode(self, text: str, context_length: int = CONTEXT_LENGTH) -> list[int]:
        """Ids con BOS/EOS y relleno con EOS hasta ``context_length``.

        El relleno es con ``<|endoftext|>`` (igual que CLIP) y no afecta a la
        salida del encoder: la atencion es causal y el embedding final se toma en
        la posicion del primer EOS.
        """
        ids = [BOS_TOKEN_ID, *self.encode_ids(text), EOS_TOKEN_ID]
        if len(ids) > context_length:
            ids = ids[: context_length - 1] + [EOS_TOKEN_ID]
        return ids + [EOS_TOKEN_ID] * (context_length - len(ids))

    def encode_batch(self, texts: list[str], context_length: int = CONTEXT_LENGTH) -> list[list[int]]:
        return [self.encode(text, context_length) for text in texts]


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    """Pares de simbolos adyacentes presentes en la palabra."""
    return set(zip(word, word[1:]))


@lru_cache(maxsize=1)
def _cached_default_tokenizer(path: str) -> ClipTokenizer:
    return ClipTokenizer(path)


def load_tokenizer(tokenizer_path: str | Path) -> ClipTokenizer:
    """Carga (con cache) el tokenizador de una ruta concreta."""
    return _cached_default_tokenizer(str(Path(tokenizer_path).resolve()))
