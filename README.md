# Zeitenwende-Projekt: SGNS-Embeddings

Eine Pipeline zur Verarbeitung von PDF-Dateien von Zeitungen aus dem Bestand der DNB und Training
von Embeddings für verschiedene Zeitperioden (jährlich, quartalsweise).

## Vorgehen

1. Korpusaufbau (je Quelle)
   - Datendownload
   - PDF-Text-Extraktion
   - Output: Rohdaten (Parquet)
2. Vorverarbeitung mit Stanza oder simpleren (weniger rechenintensiven) Verfahren
   - Tokenisierung
   - Lemmatisierung
   - POS-Tagging
   - Satzgrenzen
   - Output: Vorverarbeitete Daten (Parquet)
3. Analyse I
   - Frequenzen der Zielwörter je Zeitscheibe und Quelle überprüfen
   - Je nach Distribution Periodisierung wählen
4. Vorbereitung für Training
   - Evtl. Sampling (Frequenzen der Zielwörter normalisieren)
   - Splitten in Zeitscheiben
   - Als `corpus_file` speichern
5. Training (SGNS mit gensim Word2vec)
   - Embeddings trainieren
     1. Ein Modell je Quelle und Zeitscheibe
     2. Ein Modell über alle Quellen je Zeitscheibe
   - Output: Modelle je Zeitscheibe(/Quelle)
6. Procrustes-Alignment
   - Analog zu 5.
   - Output:
     - Ein Gesamtmodell
     - Ein Modell je Quelle
7. Analyse II
   - Qualitative Sichtung der Vektoren, Nachbarn von Zielwörtern, Clustering
   - Visualisierung  
