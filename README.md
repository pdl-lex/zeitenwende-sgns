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
   - Entfernen weicher Trennzeichen (U+00AD)
   - Tokenisierung, Satzgrenzen und POS-Tagging mit spaCy (`de_core_news_sm`); Bindestrich-Komposita
     (z. B. „NATO-Partner“) bleiben ein Token
   - Lemmatisierung mit simplemma; alte ß-Schreibung in Lemmata (Rußland, Ausschuß) wird korrigiert
   - Lemma-Overrides aus `resources/lemma_overrides.tsv` (Spalten `form`, `lemma_corrected`), um
     alle Formen der Zielwörter sicher auf das Zielwort zurückzuführen. Sie ergänzen das Wörterbuch
     von simplemma und greifen so auch in Bindestrich-Komposita (EU-Verbündeten). Kandidaten
     erzeugt `targets.override_candidates` (siehe [notebooks/analyze_preprocessed.ipynb](notebooks/analyze_preprocessed.ipynb))
   - Output: Vorverarbeitete Daten (Parquet-Shards in `data/preprocessed/<source>/`)
   - Laufzeit: siehe [Laufzeit der Vorverarbeitung](#laufzeit-der-vorverarbeitung)
3. Analyse I ([notebooks/](notebooks/))
   - Frequenzen der Zielwörter je Zeitscheibe und Quelle überprüfen
   - Je nach Distribution Periodisierung wählen
4. Filtern und Sampling ([src/zeitenwende_sgns/split.py](src/zeitenwende_sgns/split.py))
   - Evtl. Sampling (Frequenzen der Zielwörter normalisieren)
   - Splitten in Zeitscheiben (`period`: Jahre, Halbjahre oder Quartale)
   - Lemmata ohne Satzzeichen (UPOS `PUNCT`, `SYM`); Stoppwörter bleiben erhalten, da Word2Vec
     häufige Wörter ohnehin heruntersamplet (`FUNCTION_WORD_UPOS` zusätzlich ausschließen für
     eine Variante ohne Stoppwörter)
   - Output: `corpus_file`s für gensim (ein Satz pro Zeile) in
     `data/corpus/<source>/<period>/<label>.txt`, z. B. `2022.txt`, `2022-H1.txt`, `2022-Q1.txt`
5. Training ([src/zeitenwende_sgns/train.py](src/zeitenwende_sgns/train.py))
   - Embeddings trainieren (SGNS mit gensim Word2Vec)
     1. Ein Modell je Quelle und Zeitscheibe
     2. Ein Modell über alle Quellen je Zeitscheibe (Korpora vorher mit `split.combine` zusammenführen)
   - Hyperparameter nach Hamilton et al. (2016): 300 Dimensionen, Fenster 4, 5 Negative Samples,
     Subsampling 1e-5, Mindestfrequenz 5, 5 Epochen
   - Output: Wortvektoren (`KeyedVectors`) in `data/models/<source>/<period>/<label>.kv`
6. Procrustes-Alignment ([src/zeitenwende_sgns/align.py](src/zeitenwende_sgns/align.py))
   - Je Subkorpus (analog zu 5.)
   - Orthogonales Procrustes nach Hamilton et al. (2016): Vektoren auf Einheitslänge normieren,
     dann jede Zeitscheibe der Reihe nach auf die bereits alignierte vorherige rotieren (Ankerwörter
     sind das gemeinsame Vokabular; Wörter, die nur in einer Scheibe vorkommen, werden mitrotiert)
   - Output:
     - Ein aligniertes Gesamtmodell über alle Quellen
     - Ein aligniertes Modell je Quelle
     - Wortvektoren (`KeyedVectors`) in `data/aligned/<source>/<period>/<label>.kv`
7. Analyse II ([notebooks/analyze_aligned.ipynb](notebooks/analyze_aligned.ipynb))
   - Qualitative Sichtung der Vektoren, Nachbarn von Zielwörtern, Clustering
   - Nächste Nachbarn je Zeitscheibe ([src/zeitenwende_sgns/neighbors.py](src/zeitenwende_sgns/neighbors.py))
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
