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
   - Zahlen werden maskiert, damit ihre Nachbarn nicht von einzelnen, je Zeitscheibe wechselnden
     Zahlen dominiert werden: Jahreszahlen (1800–2099) werden zu `<year>`, andere Zahlen (3, 3,5,
     1.000, 1.) zu `<num>`; Komposita wie „90er-Jahre“ bleiben erhalten (`keep_numbers=True` für
     eine Variante ohne Maskierung)
   - Output: `corpus_file`s für gensim (ein Satz pro Zeile) in
     `data/corpus/<source>/<period>/<label>.txt`, z. B. `2022.txt`, `2022-H1.txt`, `2022-Q1.txt`
5. Training ([src/zeitenwende_sgns/train.py](src/zeitenwende_sgns/train.py))
   - Embeddings trainieren (SGNS mit gensim Word2Vec)
     1. Ein Modell je Quelle und Zeitscheibe
     2. Ein Modell über alle Quellen je Zeitscheibe (Korpora vorher mit `split.combine` zusammenführen)
   - Hyperparameter nach Hamilton et al. (2016): 300 Dimensionen, Fenster 4, 5 Negative Samples,
     Mindestfrequenz 5
   - Abweichend davon Subsampling 1e-4 statt 1e-5 und 10 statt 5 Epochen: Mit Hamiltons Werten
     verwirft das Subsampling auf einem Quartal (ca. 750.000 Tokens) über 80 % der Tokens, und die
     Vektoren bleiben nahezu ununterscheidbar (mittlere Kosinus-Ähnlichkeit zufälliger Wortpaare
     ≈ 1,0; mit den neuen Werten ≈ 0,65)
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
   - Diachrone Ähnlichkeitsmatrizen nach Kiyama et al. (2025): Kosinus-Ähnlichkeit der Vektoren
     eines Wortes zwischen allen Paaren von Zeitscheiben
   - Visualisierung ([src/zeitenwende_sgns](src/zeitenwende_sgns))

## Ausführung

Umgebung einrichten:

```sh
uv sync
```

Pipeline für eine Quelle in Reihenfolge ausführen (aus dem Projektverzeichnis):

```sh
uv run python -m zeitenwende_sgns.collect_data.cpp_bt  # 1. Korpusaufbau (einmalig)
uv run python -m zeitenwende_sgns.preprocess           # 2. Vorverarbeitung
uv run python -m zeitenwende_sgns.split                # 4. Splitten in Zeitscheiben
uv run python -m zeitenwende_sgns.train                # 5. Training
uv run python -m zeitenwende_sgns.align                # 6. Procrustes-Alignment
```

Quelle (`SOURCE`) und Periodisierung (`PERIOD`) sind im `__main__`-Block des jeweiligen Moduls
festgelegt (derzeit `cpp_bt` und `quarter`) und müssen in allen Schritten übereinstimmen. Die
Analysen (3. und 7.) laufen in den Notebooks in [notebooks/](notebooks/).

## Betrieb ohne Internetzugang

Auf den Rechnern mit den urheberrechtlich geschützten Daten gibt es kein Internet. Die Einrichtung
erfolgt vorher mit Internetzugang, ggf. zusammen mit
[extraceval](https://github.com/fsimonjetz/extraceval) für die PDF-Extraktion. Alles liegt an
gemeinsamen Orten, damit es nicht im Home-Verzeichnis des einrichtenden Nutzers landet.

Die Schritte 1–6 erledigt [scripts/setup_offline.sh](scripts/setup_offline.sh) (Linux, als root,
mit Internetzugang; `git` und `curl` müssen installiert sein):

```sh
curl -LsSf https://raw.githubusercontent.com/pdl-lex/zeitenwende-sgns/main/scripts/setup_offline.sh \
  | sudo bash -s -- <gruppe> [/opt]
```

Das Skript lässt sich erneut ausführen, um beide Repositorys zu aktualisieren. Von Hand:

1. Umgebungsvariablen für alle Nutzer setzen, z. B. in `/etc/profile.d/zeitenwende.sh`:

   ```sh
   export UV_PYTHON_INSTALL_DIR=/opt/uv/python  # von uv installiertes Python
   export HF_HOME=/opt/hf-cache                 # Modelle von Docling (extraceval)
   ```

2. `git` und `uv` systemweit installieren:

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
   ```

3. Repositorys am endgültigen Ort klonen (z. B. `/opt/zeitenwende-sgns`, `/opt/extraceval`) und
   jeweils `uv sync` ausführen (`.venv` nicht nachträglich verschieben oder kopieren, es enthält
   absolute Pfade). Ein privates Repository lässt sich als Datei übergeben
   (`git bundle create extraceval.bundle --all`, dann `git clone extraceval.bundle`)
4. In `extraceval` die Modelle laden: `uv run extraceval init` (alle Tools müssen `ok` melden)
5. Rechte für eine gemeinsame Gruppe setzen:

   ```sh
   chgrp -R <gruppe> /opt/zeitenwende-sgns /opt/extraceval /opt/hf-cache
   chmod -R g+rwX /opt/zeitenwende-sgns /opt/extraceval /opt/hf-cache
   chmod -R a+rX /opt/uv
   ```

6. Offline-Betrieb in `/etc/profile.d/zeitenwende.sh` ergänzen:

   ```sh
   export UV_OFFLINE=1 UV_NO_SYNC=1 HF_HUB_OFFLINE=1
   ```

   `uv run` prüft sonst bei jedem Aufruf die Umgebung und versucht, Pakete nachzuladen, z. B. die
   per URL eingebundene Abhängigkeit `de_core_news_sm` (GitHub) oder Teile von PyTorch.
7. Notebook-Oberfläche bereitstellen (JupyterLab oder VS Code mit Python- und Jupyter-Erweiterung)
8. Rohdaten nach `data/raw/<source>/` kopieren (der Download in `collect_data` funktioniert offline
   nicht)

Offline lässt sich der Code unter `src/` beliebig ändern; neue Abhängigkeiten (`uv add`) oder
andere Änderungen an den Abhängigkeiten in `pyproject.toml` sind dagegen nicht möglich. Alle
benötigten Pakete müssen also vor der Einrichtung im Projekt stehen.

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
