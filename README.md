# Zeitenwende-Projekt: SGNS-Embeddings

Eine Pipeline zur Verarbeitung von PDF-Dateien von Zeitungen aus dem Bestand der DNB und Training
von Embeddings für verschiedene Zeitperioden (jährlich, quartalsweise).

## Vorgehen

Eingabe ist neben den Korpora die Liste der Zielwörter in
[resources/zielwoerter.txt](resources/zielwoerter.txt) (siehe [resources/README.md](resources/README.md)).

1. Korpusaufbau (je Quelle; [src/zeitenwende_sgns/collect_data/](src/zeitenwende_sgns/collect_data/))
   - Ggf. Download (für Testdaten)
   - Vorauswahl relevanter Jahre
   - PDF-Text-Extraktion
     - Parallel: Evaluation des Outputs verschiedener Extraktions-Bibliotheken
   - Output: Rohdaten (Parquet) in `data/raw/<source>` mit Datum, Dokument-ID (Hash), Rohtext
2. Vorverarbeitung ([src/zeitenwende_sgns/preprocess.py](src/zeitenwende_sgns/preprocess.py))
   - Filterung
     - Deduplikation (soweit möglich)
     - Entfernen extrem kurzer/langer Dokumente
   - Tokenisierung, Satzgrenzen und POS-Tagging mit spaCy (`de_core_news_sm`); Bindestrich-Komposita
     (z. B. „NATO-Partner“) bleiben ein Token
   - Lemmatisierung mit simplemma; alte ß-Schreibung in Lemmata (Rußland, Ausschuß) wird korrigiert
   - Lemma-Overrides aus `resources/lemma_overrides.tsv` (Spalten `form`, `lemma_corrected`), um
     alle Formen der Zielwörter sicher auf das Zielwort zurückzuführen. Einen Entwurf erzeugt
     `targets.draft_overrides` (siehe [notebooks/analyze_preprocessed.ipynb](notebooks/analyze_preprocessed.ipynb))
   - Output: Vorverarbeitete Daten (Parquet-Shards in `data/preprocessed/<source>/`)
   - Laufzeit: siehe [Laufzeit der Vorverarbeitung](#laufzeit-der-vorverarbeitung)
3. Analyse I ([notebooks/](notebooks/))
   - Frequenzen der Zielwörter je Zeitscheibe und Quelle überprüfen
   - Je nach Distribution Periodisierung wählen
4. Filtern und Sampling ([src/zeitenwende_sgns/split.py](src/zeitenwende_sgns/split.py))
   - Evtl. Sampling (Frequenzen der Zielwörter normalisieren)
   - Splitten in Zeitscheiben
   - Als `corpus_file`s speichern
5. Training ([src/zeitenwende_sgns/train.py](src/zeitenwende_sgns/train.py))
   - Embeddings trainieren (SGNS mit gensim Word2Vec)
     1. Ein Modell je Quelle und Zeitscheibe
     2. Ein Modell über alle Quellen je Zeitscheibe
   - Output: Modelle je Zeitscheibe(/Quelle)
6. Procrustes-Alignment ([src/zeitenwende_sgns/align.py](src/zeitenwende_sgns/align.py))
   - Je Subkorpus (analog zu 5.)
   - Output:
     - Ein aligniertes Gesamtmodell über alle Quellen
     - Ein aligniertes Modell je Quelle
7. Analyse II ([notebooks/](notebooks/))
   - Qualitative Sichtung der Vektoren, Nachbarn von Zielwörtern, Clustering
   - Visualisierung ([src/zeitenwende_sgns](src/zeitenwende_sgns))

## Laufzeit der Vorverarbeitung

Referenzlauf: `cpp_bt` (Bundestagsreden, 77.517 Dokumente, 307 Mio. Zeichen, 50,4 Mio. Tokens)
auf einem Apple M3 (8 Kerne, 24 GB RAM), 8 Worker. Dauer: ca. 5 Minuten (mit Stanza statt
spaCy/simplemma: ca. 66 Minuten).

Faustregeln:

- 1 GB Rohtext ≈ 17 min
- 100 Mio. Tokens ≈ 10 min

Für eine Schätzung die Zeichenzahl des Rohkorpus durch den Referenzdurchsatz (ca. 58 Mio. Zeichen
pro Minute) teilen:

```python
import polars as pl

n_chars = (
    pl.scan_parquet("data/raw/<source>/<source>.parquet")
    .select(pl.col("text").str.len_chars().sum())
    .collect()
    .item()
)
print(f"{n_chars / 58e6:.0f} min")
```

Unterbrochene Läufe können einfach neu gestartet werden; fertige Shards werden übersprungen.
