# Projektbeschreibung: Ticket-Clustering mit SBERT + HDBSCAN (Offline)

## Ziel
Entwicklung eines lokalen Python-Tools, das große Mengen von Störungstexten (Tickets) semantisch clustert und eine **Übersichts-Tabelle (Dashboard)** als Excel-Report erzeugt.  
Die Detailebene (jede einzelne Zeile) wird **nicht** benötigt – es geht um eine schnelle Übersicht über Themen und deren Häufigkeiten.

## Input
- **Dateiformat**: Excel oder CSV  
- **Spalten**:
  - `line_nr`: fortlaufende Zeilennummer (optional, für Nachvollziehbarkeit)
  - `text`: Originaltext der Meldung
  - `occurrences`: Anzahl der Vorkommen (z. B. 1–1000, da 500k → 50k Uniques reduziert)

Beispiel:
| line_nr | text                                              | occurrences |
|---------|--------------------------------------------------|-------------|
| 1       | Automatische Anmeldung mit PKI geht nicht         | 120         |
| 2       | Statt Login mit PKI kommt Anmeldemaske            | 95          |
| 3       | SNC Fehler / Anmeldeproblem mit PKI               | 70          |
| 4       | Kennzahlen-Dashboard OT – keine Einwahl möglich   | 50          |

## Output
- **Excel-Datei**: `overview.xlsx` mit einem **Sheet** `Overview`
- **Inhalte**:
  - Anzahl Input-Zeilen (z. B. 500,000)
  - Anzahl Unique-Zeilen (z. B. 50,000)
  - Anzahl Cluster (ohne Noise)
  - Anzahl Noise-Zeilen (Cluster = -1)
  - Größter Cluster (Anzahl Zeilen + Anteil in %)
  - Kleinster Cluster (Anzahl Zeilen)
  - Median Clustergröße
  - Top-10 Cluster (nach Gesamtvorkommen):
    - `cluster_id`
    - `label`
    - `count_total` (Summe occurrences aller Mitglieder)
    - `share_%` (count_total / input_rows_total)

Optional: kleines Balkendiagramm der Top-10 Cluster.

## Vorgehen / Architektur
1. **Preprocessing**
   - Normalisierung (Lowercasing, Unicode-NFKC, Whitespace, Satzzeichen vereinheitlichen)
   - Stopwörter entfernen (aber Negationen wie *nicht/kein* behalten)
   - Fachbegriffe (PKI, SNC, APEX, SAP GUI, Dashboard, etc.) explizit beibehalten
   - Hash-Map: `normalized_text → [occurrences, original_text]`

2. **Embedding**
   - Modell: `paraphrase-multilingual-MiniLM-L12-v2` (lokal gecached)
   - Batch-Encoding mit 8 CPU-Kernen parallel
   - Vektorformat: `float32`, NumPy Array

3. **Clustering**
   - Optional: UMAP (15 Dimensionen, cosine → euclidean)
   - HDBSCAN:
     - `min_cluster_size`: 25 (tuneable)
     - `metric`: euclidean (nach UMAP)
   - Ergebnis: `labels` für jeden Unique-Text

4. **Cluster-Labeling**
   - Keyword-basiert (PKI, SNC, Anmeldung, Dashboard, Bug, …)
   - YAKE Keyphrase-Extraktion als Fallback
   - Kürze Label auf max. 60 Zeichen

5. **Aggregation**
   - Für jedes Cluster: Summe über `occurrences`
   - Kennzahlen berechnen:
     - `count_total_lines` (Summe aller occurrences im Cluster)
     - `count_unique_texts` (Unique-Mitglieder im Cluster)
     - `share_%` = count_total_lines / input_rows_total

6. **Reporting**
   - Schreibe `overview.xlsx`
   - Sheet `Overview`:
     - KPIs
     - Tabelle Top-10 Cluster
     - Diagramm Top-10 Cluster (optional)

## Performance / Ressourcen
- Input: 50k Uniques → Embeddings (384 Dimensionen)
- Speicherbedarf: ~75–150 MB RAM
- Laufzeit:
  - Embedding: 5–10 Minuten (parallelisiert auf 8 Kerne eines MacBook Pro M1/M2 12-core)
  - UMAP: Sekunden bis wenige Minuten
  - HDBSCAN: Sekunden
- Gesamtlaufzeit: typischerweise <15 Minuten

## Konfiguration
Parameter steuerbar über `config.yaml`:
```yaml
embedding:
  model_name: paraphrase-multilingual-MiniLM-L12-v2
  batch_size: 256
  n_jobs: 8

umap:
  enabled: true
  n_neighbors: 15
  min_dist: 0.0
  n_components: 15

hdbscan:
  min_cluster_size: 25
  min_samples: null
  cluster_selection_epsilon: 0.0

labeling:
  keywords: ["PKI","SNC","APEX","SAP","GUI","Login","Anmeldung","Dashboard","Bug","Fehler"]
  yake_topk: 10
  yake_max_ngram: 3

output:
  top_n_clusters: 10
  include_noise: false
