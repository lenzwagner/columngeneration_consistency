# Column Generation for Nurse Rostering

Ein Optimierungssystem zur Erstellung von Dienstplänen für Pflegepersonal basierend auf Column Generation.

## Projektbeschreibung

Dieses Projekt implementiert einen Column Generation Algorithmus zur Lösung von Nurse Rostering Problems (NRP). Das System optimiert die Dienstplanerstellung unter Berücksichtigung verschiedener Constraints wie Schichtwechsel, Arbeitstage, Ruhetage und Leistungsfähigkeit des Personals.

## Hauptkomponenten

### Kernmodule

- **masterproblem.py**: Implementierung des Master-Problems mit Gurobi-Optimierung
- **subproblem.py**: Pricing-Problem zur Generierung neuer Dienstplanmuster
- **loop.py**: Hauptskript zur Ausführung des Column Generation Algorithmus

### Lösungsansätze

- **cg_naive.py**: Naive Column Generation Implementierung
- **cg_behavior.py**: Verhaltensbasierte Column Generation Variante

### Utility-Module

- **Utils/setup.py**: Konfiguration und Setup
- **Utils/gcutil.py**: Hilfsfunktionen für Column Generation
- **Utils/compactsolver.py**: Kompaktes Lösungsverfahren
- **Utils/demand.py**: Nachfrage-Verwaltung
- **Utils/plot.py**: Visualisierung der Ergebnisse
- **Utils/aggundercover.py**: Aggregation von Unterdeckungen

## Anforderungen

- Python 3.x
- Gurobi Optimizer (Lizenz erforderlich)
- NumPy
- Pandas
- Matplotlib/Plotly (für Visualisierungen)

## Installation

1. Repository klonen:
```bash
git clone <repository-url>
cd columngeneration
```

2. Abhängigkeiten installieren:
```bash
pip install numpy pandas matplotlib gurobipy openpyxl
```

3. Gurobi-Lizenz konfigurieren

## Verwendung

### Grundlegende Ausführung

```bash
python loop.py
```

### Parameter-Konfiguration

Die wichtigsten Parameter können in `loop.py` angepasst werden:

- `eps_ls`: Epsilon-Werte für Leistungsabnahme
- `chi_ls`: Chi-Werte für Schichtwechsel-Constraints
- `time_Limit`: Zeitlimit für Optimierung (Sekunden)
- `max_itr`: Maximale Anzahl Column Generation Iterationen

## Dateneingabe

Eingabedaten werden aus Excel-Dateien im `data/`-Verzeichnis geladen:

- **data_demand.xlsx**: Nachfragemuster für verschiedene Schichten
- **data.xlsx**: Stammdaten für Mitarbeiter und Schichten

## Ausgabe

Die Ergebnisse werden gespeichert in:

- `results/`: CSV- und Excel-Dateien mit Optimierungsergebnissen
- `images/schedules/`: Visualisierungen der generierten Dienstpläne

## Algorithmus

Das System verwendet einen klassischen Column Generation Ansatz:

1. Lösung des relaxierten Master-Problems
2. Extraktion der Dual-Werte
3. Lösung der Pricing-Probleme (Subprobleme)
4. Hinzufügen von Spalten mit negativen reduzierten Kosten
5. Wiederholung bis zur Konvergenz
6. Finale Lösung mit Integer-Variablen

## Constraints

Das System berücksichtigt:

- Nachfragedeckung pro Schicht und Tag
- Mindest- und Maximalanzahl aufeinanderfolgender Arbeitstage
- Ruhetage zwischen Arbeitsblöcken
- Verbotene Schichtfolgen
- Leistungsabnahme durch konsekutive Schichten
- Konsistenz der Dienstpläne

## Metriken

Folgende Kennzahlen werden berechnet:

- Undercoverage: Gesamte Unterdeckung der Nachfrage
- Understaffing: Unterdeckung durch fehlende Mitarbeiter
- Performance Loss: Leistungsverlust durch Ermüdung
- Consistency: Konsistenz der Schichtzuordnung
- Autocorrelation: Autokorrelation der Schichtwechsel

## Struktur

```
columngeneration/
├── masterproblem.py       # Master-Problem
├── subproblem.py          # Pricing-Problem
├── loop.py                # Hauptausführung
├── cg_naive.py            # Naive Variante
├── cg_behavior.py         # Verhaltensbasierte Variante
├── data/                  # Eingabedaten
├── results/               # Ergebnisse
├── images/                # Visualisierungen
└── Utils/                 # Hilfsfunktionen
```

## Lizenz

Dieses Projekt dient zu Forschungszwecken.

## Kontakt

Für Fragen oder Anmerkungen bitte ein Issue erstellen.
