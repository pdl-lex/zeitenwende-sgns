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
   - Output: Vorverarbeitete Daten (Parquet)
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
