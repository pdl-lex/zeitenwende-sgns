# Ressourcen

## `zielwoerter.txt`

Liste der Zielwörter, die die Analyse verwendet (UTF-8, ein Zielwort je Zeile in Lemmaform). Sie
ist aus der Spalte `Zielwort` der Masterliste abgeleitet; Grundformen werden hier korrigiert.

## `master_zielwoerter.tsv`

Masterliste der Zielwörter mit Bewertungen (UTF-8, tabulatorgetrennt, eine Zeile je Zielwort).
Dezimaltrennzeichen ist der Punkt; Markierungsspalten enthalten `1` oder bleiben leer.

| Spalte | Bedeutung |
| --- | --- |
| `Zielwort` | Zielwort (Lemma) |
| `Quelle: IN`, `Quelle: WB`, `Quelle: AP` | Wort wurde von dieser Quelle vorgeschlagen |
| `Quellen gesamt` | Anzahl der Quellen |
| `Relevanz FS`, `Relevanz WH` | Relevanzbewertung (1–5) je Bewerter:in |
| `Relevanz-Differenz` | Betrag der Differenz beider Bewertungen |
| `Relevanz gesamt` | Mittelwert beider Bewertungen |
| `Imagewandel` | Ist zu erwarten, dass das Wort einen Imagewandel wie vermutet (Bundeswehr) aufweist? (in beide Richtungen, besser oder schlechter werdend) |
| `Multiwort` | Ist zu erwarten, dass dieses vielseitig verwendbare Wort vermehrt im Zeitenwende-Diskurs auftaucht? |
| `Neologismus` | Ist zu erwarten, dass das Wort erstmalig oder mit massiv erhöhter Frequenz auftaucht? |
| `Fremddiskurs` | Gehört das Wort zu einem anderen Diskurs, könnte aber dennoch interessant sein? |
| `Niedrige Frequenz` | Ist zu befürchten, dass das Wort zu selten auftaucht? |
| `Klärung` | Welchen Erkenntnisgewinn erwarten wir bei diesem Wort? |
| `Kumulierte Relevanz` | Gesamtrelevanz |
| `Alternativen und Flektionen` | Varianten des Zielworts, getrennt durch `\|` |

## `lemma_overrides.tsv`

Optionale Korrekturen der Lemmatisierung (UTF-8, tabulatorgetrennt). Die Vorverarbeitung ordnet
jeder Wortform in `form` das Lemma in `lemma_corrected` zu. `lemma_predicted` (das Lemma ohne
Korrektur) und `count` (Häufigkeit der Form) dienen nur der Dokumentation. Einen Entwurf erzeugt
`targets.draft_overrides` (siehe [notebooks/analyze_preprocessed.ipynb](../notebooks/analyze_preprocessed.ipynb)).
