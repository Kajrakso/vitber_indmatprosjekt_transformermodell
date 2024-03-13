# vitber_indmatprosjekt_transformermodell

13.03.2024

---

## Hensikt

*Kopiert fra prosjektbeskrivelsen.*

Målet med dette prosjektet er å forstå hvordan dyp læring (deep learning) fungerer.
Mer spesifikt skal vi implementere transformermodellen som er av hovedkomponentene
i store språkmodeller som ChatGPT.

Testproblemene vi skal studere er å sortere en liste med tall og addere to heltall.
Vi skal se hvordan vi kan formulere dette som et problem der målet er å predikere
neste heltall i en sekvens. Dette er selvsagt en utrolig tungvint måte å implementere
sortering og addisjon på. Poenget er å se hvordan ulike problemer kan struktureres som
prediksjoner av heltall og dette gir en god forståelse av språkmodeller siden de fungerer
ut i fra samme prinsipp. Prosjektet har også en frivillig bonusoppgave der vi modellerer
språk ved å predikere neste bokstav i en setning.

## Innlevering:

Oversikt over filer som skal leveres inn samt en beskrivelse av innholdet.

- `src/`: kildekode
    - `handin.ipynb`: hovednotebook for prosjektet

    - `neural_network.py`: Klassen `NeuralNetwork`
    - `layers.py`: `SoftMax`, `Attention`, `CrossEntropy`, `LinearLayer`, `ReLu`, `EmbedPosition` and `FeedForward`
    - `layers_numba.py`: numbaifisert versjon av `layers.py`
    - `utils.py`: inneholder `onehot()`
    - `utils_numba.py`: numbaifisert versjon av `utils.py`

    - `data_generators.py`: har funksjonar til å generera trenings- og testdata.
    - `text_generation.py`: genererer tekst.
    - `test_implementation.ipynb`: notebook med diverse tester

    - `train_network.py`: trener nettverket på treningsdata
    - `test_network.py`: tester nettverket på testdata
    - `train_test_params.py`: dataklasser med trenings- og testdata

    - `input.txt`: Utlevert Shakespeare-tekst
    - `shakespeare.txt`: Shakespeare samlede verk

    - `*_dumps/`: nettverk trent over $1000$+ iterasjoner.
- `Pipfile`: TOML fil for lage et virtuelt miljø med Pipenv
- `Pipfile.lock`: Låser nøyaktige versjoner av pakker brukt i det virtuelle miljøet. Installeres med kommandoen `pipenv sync`.
