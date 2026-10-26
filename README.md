# Zeitenwende-Projekt: SGNS-Embeddings

Eine Pipeline zur Verarbeitung von PDF-Dateien von Zeitungen aus dem Bestand der DNB und Training
von Embeddings für verschiedene Zeitperioden (jährlich, quartalsweise).

## Vorgehen

1. Korpusaufbau (je Quelle; [src/zeitenwende_sgns/collect_data/](src/zeitenwende_sgns/collect_data/))
   - Ggf. Download (für Testdaten)
   - Vorauswahl relevanter Jahre
   - PDF-Text-Extraktion
     - Parallel: Evaluation des Outputs verschiedener Extraktions-Bibliotheken
   - Output: Rohdaten (Parquet) in `data/raw/<source>` mit Datum, Dokument-ID (Hash), Rohtext
2. Vorverarbeitung ([src/preprocess.py](src/preprocess.py))
   - Filterung
     - Deduplikation (soweit möglich)
     - Entfernen extrem kurzer/langer Dokumente
   - Stanza o.ä.
     - Tokenisierung
     - Lemmatisierung
     - POS-Tagging
     - Satzgrenzen
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

Referenzlauf: `cpp_bt` (Bundestagsreden, 77.517 Dokumente, 307 Mio. Zeichen, 51,5 Mio. Tokens)
auf einem Apple M3 (8 Kerne, 24 GB RAM), 8 Worker, nur CPU. Dauer: ca. 66 Minuten.

| Durchsatz pro Stunde | |
|---|---|
| Tokens (Stanza-Wörter) | ~46 Mio. |
| Zeichen | ~274 Mio. |
| Bytes (UTF-8) | ~280 MB |

Faustregeln:

- 1 GB Rohtext ≈ 3,6 h
- 100 Mio. Tokens ≈ 2,2 h
- Ca. 6 Zeichen pro Token (deutscher Text)

Für die Schätzung Zeichen oder Bytes verwenden, nicht die Anzahl der Dokumente: Die Dauer hängt
von der Textlänge ab, und die Dokumentlängen schwanken stark (im Referenzlauf zwischen ca. 10 und
31 Dokumenten/s).

### ETA schätzen

1. Zeichenzahl des Rohkorpus bestimmen und durch den Referenzdurchsatz teilen:

   ```python
   import polars as pl

   n_chars = (
       pl.scan_parquet("data/raw/<source>/<source>.parquet")
       .select(pl.col("text").str.len_chars().sum())
       .collect()
       .item()
   )
   print(f"{n_chars / 274e6:.1f} h")
   ```

2. Bei anderer Hardware oder anderer Textsorte (z. B. Zeitungstext aus PDFs mit OCR-Rauschen)
   den Durchsatz an einer Stichprobe neu messen, z. B. `preprocess(..., limit=2000)` mit
   eigenem `output_dir`. Die ersten Minuten (Laden der Modelle) nicht mitrechnen. Der Durchsatz
   skaliert ungefähr linear mit der Anzahl der CPU-Kerne.

Unterbrochene Läufe können einfach neu gestartet werden; fertige Shards werden übersprungen.
