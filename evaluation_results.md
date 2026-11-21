# Wyniki ewaluacji: tydzień 1

Przegląd z 2026-11-20, eksperyment `9c8e117`. [PR #74](https://github.com/wiertmir/mini-rag/pull/74), [podsumowanie tygodnia #21](https://github.com/wiertmir/mini-rag/issues/21).

## Co z tego wynika

Do dalszej pracy lokalnej wybrałbym **Gemma 4 12B Q8 z kontekstem 8K**.
Z oboma modelami embeddingowymi uzyskała 25/28 trafień w oczekiwany dokument.
Większe modele nie poprawiły tego wyniku, a odpowiedzi trwały dłużej.
Qwen 4B jest rozsądnym, lżejszym punktem wyjścia dla embeddingów.
OpenAI (`text-embedding-3-small` + `gpt-5.4-mini`) zostaje wersją porównawczą:
ten sam wynik 25/28, średnio 2,66 s na pytanie.

**25/28 nie oznacza 25 poprawnych odpowiedzi.** Liczymy, czy odpowiedź cytuje
przynajmniej jeden oczekiwany dokument. To niedoskonała miara: dobra odpowiedź
z innego rozdziału może dostać zero, a odpowiedź oparta na samej bibliografii
może dostać punkt. Najpierw poprawiłbym ocenę źródeł, potem porównywał modele.

## Wyniki odpowiedzi

Sprawdziłem siedem przebiegów, łącznie 210 rekordów. Każdy obejmuje 28 pytań
z odpowiedzią w korpusie i dwa spoza korpusu (Q29 i Q30).
„Citation hits” to opisane wyżej trafienia w oczekiwany dokument.
„Answerable refusals” liczy odmowy tylko dla pierwszych 28 pytań.
Błąd wykonania też zostaje w mianowniku 28.

| Embeddings | Answer model / records | Citation hits | Answerable refusals | Errors | Mean s | Median s | Max s |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-embedding:4b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding_4b/gemma4_12b-q8-8k/baseline.jsonl) | 25/28 | 2 | 0 | 35.99 | 34.56 | 75.80 |
| qwen3-embedding:4b | [gemma4:31b](eval/results/qwen3-embedding_4b/gemma4_31b/baseline.jsonl) | 25/28 | 2 | 0 | 81.36 | 78.15 | 134.99 |
| qwen3-embedding:4b | [qwen3.6:35b](eval/results/qwen3-embedding_4b/qwen3.6_35b/baseline.jsonl) | 24/28 | 2 | 0 | 41.27 | 41.16 | 65.72 |
| qwen3-embedding:8b | [gemma4:12b-q8-8k](eval/results/qwen3-embedding_8b/gemma4_12b-q8-8k/baseline.jsonl) | 25/28 | 1 | 0 | 34.04 | 31.89 | 72.30 |
| qwen3-embedding:8b | [gemma4:31b](eval/results/qwen3-embedding_8b/gemma4_31b/baseline.jsonl) | 22/28 | 2 | 0 | 90.93 | 89.09 | 158.87 |
| qwen3-embedding:8b | [qwen3.6:35b](eval/results/qwen3-embedding_8b/qwen3.6_35b/baseline.jsonl) | 23/28 | 2 | 1 | 64.61 | 58.67 | 118.42 |
| text-embedding-3-small | [gpt-5.4-mini](eval/results/text-embedding-3-small/gpt-5.4-mini/baseline.jsonl) | 25/28 | 0 | 0 | 2.66 | 2.55 | 4.19 |

Wszystkie modele prawidłowo odmówiły odpowiedzi na Q29 i Q30, bez cytowań.
Nie było błędów `invalid_citation`. Wystąpił jeden błąd wykonania:
Q08 dla Qwen 8B + Qwen 35B zwróciło `ValidationError`. Pętla kontynuowała pracę
i zapisała wszystkie 30 rekordów. Zapisane dane nie mówią, czy błąd powstał
przy parsowaniu odpowiedzi SDK, czy przy późniejszej walidacji.

Gemma 31B miała średni czas odpowiedzi około 2,26 razy dłuższy od 12B
przy embeddingach 4B i 2,67 razy przy 8B. Przy wynikach 25/28 i 22/28
nie widzę tu powodu do wyboru większej Gemmy. Qwen 35B uzyskał 24/28 i 23/28.
Embeddingi 8B poprawiły wyszukiwanie dokumentów, ale nie wynik odpowiedzi Gemmy 12B.

Czasy obejmują całe pytanie: embedding, oczekiwanie/ładowanie modeli,
generowanie i obróbkę odpowiedzi. Nie są pomiarem szybkości samego LLM.
Na GPU 16 GB modele mogą wymagać przeładowywania lub korzystać z RAM.
Podczas osobnego sprawdzania konfiguracji widziałem wymiany modeli przy
równoległych zapytaniach Qwen; nie mamy potwierdzenia, że działo się to
w każdym z tych pomiarów. Nie zapisano kosztów tokenów ani zużycia energii.
Lokalne modele nadal mają zalety: prywatność, pracę offline i brak opłat za zapytanie.

## Czy wyszukiwarka znalazła przydatne źródła?

Sprawdziłem także trzy przebiegi samego wyszukiwania, razem 90 rekordów.
`Document hit@5` oznacza obecność oczekiwanego dokumentu w pięciu wynikach.

| Embeddings | Document hit@5 | Saved document misses |
|---|---:|---|
| text-embedding-3-small | 27/28 | Q01 |
| qwen3-embedding:4b | 26/28 | Q01, Q21 |
| qwen3-embedding:8b | 28/28 | None |

Obecność dokumentu nie wystarcza. Przykłady:

- Q18 / 4B: cztery z pięciu fragmentów to bibliografia. Piąty jest wstępem
  o lotnictwie i nie daje pełnego porównania transportu drogowego z lotniczym
  pod kątem gęstości energii. Odmowa nie musi oznaczać błędu modelu.
- Q21 / 8B: trafienie `WGIII_CH12_p043_c001` to fragment tabeli/odnośników,
  bez wyjaśnienia LCA. W pozostałych wynikach też jest bibliografia.
  Wszystkie lokalne modele odmówiły odpowiedzi.
- Q25 / 8B + Gemma 12B: wszystkie pięć fragmentów to bibliografia.
  Model ułożył wyjaśnienie na podstawie tytułów o finansowaniu i ograniczaniu
  ryzyka. Dostał punkt za cytowanie, choć tytuły nie uzasadniają całego wyjaśnienia.
  Odmowy większych modeli nie są tu dowodem ich gorszej jakości.

To temat [#76](https://github.com/wiertmir/mini-rag/issues/76).
Filtrowanie bibliografii lub reranking warto sprawdzić jako nowy eksperyment,
z zachowaniem obecnego baseline'u.

## Gdzie ocena jest niesprawiedliwa lub gubi cytowania

Trzy odpowiedzi OpenAI dostały zero za dokument, mimo że znalazłem
potwierdzenie ich treści w PDF-ach:

| Question | Labeled expectation | Actual cited evidence | Interpretation |
|---|---|---|---|
| Q01 | WGIII_CH01 | WGI_CH01, physical p.9 | Explicitly states WGIII assesses mitigation response options. Valid shorter answer. |
| Q09 | WGIII_CH05 / WGIII_CH06 | WGIII_CH11, physical p.27; also WGIII_CH08 | Efficiency enables clean electrification and reduces supply needs. |
| Q19 | WGIII_CH11 | WGIII_CH06, physical p.65 | Names cement, primary steel and chemical feedstocks as challenging to electrify. |

Poprawki oczekiwanych dokumentów są w [#75](https://github.com/wiertmir/mini-rag/issues/75).
Po przeglądzie źródeł trzeba zapisać nową wersję pytań i przeliczyć zachowane
odpowiedzi. Dotychczasowe 25/28 pozostaje wynikiem dla starej wersji etykiet.

Osobny problem dotyczy Q22 / 4B + Qwen 35B. Model użył znaczników [2, 5]
i [1, 3, 4], a parser obsługuje pojedyncze [n]. W zapisanych cytowaniach zostało
tylko odwołanie 3. Trzeba ustalić obsługę grup cytowań, zachowując oryginalny
tekst odpowiedzi: [#78](https://github.com/wiertmir/mini-rag/issues/78).
Dla błędu Q08 potrzebna jest informacja o etapie awarii i próba odtworzenia
offline: [#77](https://github.com/wiertmir/mini-rag/issues/77).

## Ręczne sprawdzenie odpowiedzi i PDF-ów

Porównałem dziesięć zapisanych odpowiedzi OpenAI z roboczymi odpowiedziami
wzorcowymi. To wybrana próbka, więc nie wyliczam z niej jakości całego zbioru.

| Question | Assessment |
|---|---|
| Q01 | Correct core purpose, supported by WGI despite the label miss. |
| Q02 | Correct mitigation/adaptation distinction, including an explicit adaptation definition. |
| Q03 | Demographic/economic/activity drivers and energy/land/process sources align with the reference. |
| Q05 | Rapid cuts and mid-century net zero; numerical timing is supported on cited pages. |
| Q09 | Correct complementary efficiency/electrification roles; label false negative. |
| Q10 | Low-carbon electricity and system integration needs align with the reference. |
| Q19 | Appropriate steel/cement/chemicals examples; label false negative. |
| Q25 | De-risking and mobilizing private capital align with the reference and substantive chapter evidence. |
| Q29 | Explicit refusal, no Apple earnings claim or citations. |
| Q30 | Explicit refusal, no invented Tokyo population or citations. |

Przejrzałem też wszystkie lokalne odpowiedzi ocenione jako nietrafione oraz
błąd Q08. Tabela w dodatku pokazuje obecność dokumentu, a nie jakość fragmentu.

Pięć wybranych twierdzeń sprawdziłem w treści PDF-ów. Numery stron poniżej
liczą fizyczne strony pliku od 1. Dla cytowania wielostronicowego sprawdziłem
obie strony i zgodność zakresu z fragmentem.

| Claim | Citation | Short source evidence | Result |
|---|---|---|---|
| Q01: WGIII's role | WGI_CH01 p.9; WGI_CH01_p009_c001 | “Working Group III (WGIII) assesses mitigation response options.” | Supported |
| Q02: adaptation adjusts to climate effects | WGIII_CH14 p.47; WGIII_CH14_p047_c002 | “adjustment to actual or expected climate and its effects” | Supported |
| Q05: immediate cuts / net-zero timing | WGIII_CH03 pp.4–5; WGIII_CH03_p004_c003 | p.4 defines immediate action; p.5 gives CO2 halving in the 2030s and net zero in the 2050s. | Supported across both pages |
| Q09: efficiency complements electrification | WGIII_CH11 p.27; WGIII_CH11_p027_c002 | “Energy efficiency is a critical strategy for net zero transitions and enabling clean electrification.” | Supported |
| Q19: hard-to-electrify industry | WGIII_CH06 p.65; WGIII_CH06_p065_c001 | Cement, primary steel and chemical feedstocks are challenging to electrify. | Supported |

Wszystkie pięć twierdzeń miało potwierdzenie. Osobno sprawdziłem zgodność
518 wpisów metadanych cytowań i wszystkich zachowanych ID fragmentów
z aktualnymi chunkami. Zgodne metadane nie gwarantują prawdziwości odpowiedzi.

## Co sprawdzono na koniec tygodnia

Tydzień 1 zaliczony; poprawki #75, #76, #77 i #78 pozostają otwarte.
Można przejść do tygodnia 2 i używać obecnego pipeline'u, schematów oraz
embeddingów jako punktu porównania. To ocena zakresu tygodnia 1, nie gotowości produkcyjnej.

- Repozytorium było czyste na początku audytu. Pracowałem w osobnym worktree
  z Pythonem 3.13.15. Struktura wymagana na dzień 1 była kompletna.
- Wszystkie 20 PDF-ów z manifestu dało się odczytać; niczego nie pobierałem.
- Dwa pełne ingesty dały po 6264 fragmenty, identyczne bajtowo z obecnym
  plikiem chunków. Trwały 334,45 s i 333,42 s. Zapisałem je w
  `/tmp/pr74-week/run1` i `/tmp/pr74-week/run2`, bez nadpisywania indeksów.
- Indeksy miały prawidłowe liczby wierszy, wymiary, hashe i parametry ingestu:
  OpenAI (6264, 1536), Qwen 4B (6264, 2560), Qwen 8B (6264, 4096).
  Wyszukanie wektora w jego własnym indeksie dawało wynik około 1.
- Testy offline bez klucza API i z zamkniętym endpointem: 313 zaliczonych,
  2 testy live pominięte przez filtr. Testy ewaluacji: 38 zaliczonych.
  Ruff (lint/format), mypy dla źródeł i zmienionych testów oraz importy przeszły.
  Testy zapisanych wyników rozszerzyłem z jednego do wszystkich siedmiu przebiegów;
  testy smoke przeliczają trafienia względem obecnych pytań.
- [Acceptance run 44738489501](https://github.com/wiertmir/mini-rag/actions/runs/44738489501):
  6 zaliczonych, 0 pominiętych na `9c8e117`, po sprawdzeniu kompletności PDF-ów.
  Obejmuje metadane, deterministyczność, obie strategie chunkingu, overlap,
  brakujący PDF i pustą stronę. Późniejsze commity zmieniały tylko testy i dokumentację.
- `.env`, indeksy, PDF-y i logi nie były śledzone przez Git; przykładowy klucz
  był placeholderem. Skan historii zwrócił jeden fałszywy alarm w tytule merge'a
  (`ask-function-and-cli-command`), bez klucza. Skan samych patchy nie zwrócił kandydatów.
- Audyt korzystał z zapisanych odpowiedzi i embeddingów. Nie wykonywałem zapytań
  generujących odpowiedzi, embeddingów pytań ani ponownego embeddingowania korpusu.
  Implementacja ucznia i wyniki nie zostały zmienione. Końcowe CI i merge są w PR;
  instrukcje odtworzenia są w README.

## Warunki pomiaru i dalsze porównania

Wyniki i statystyki przeliczyłem z rekordów. Wspólne ustawienia siedmiu przebiegów:

- 30 pytań, `top_k=5`, wykonanie sekwencyjne;
- chunking sekcyjny 600/90, minimum 150 tokenów; zgodne ustawienia tokenizera,
  pomijania podpisów i obsługi ramek;
- dataset SHA-256: `268f7b1440c1ac549a0c77d663cb0758452208ecfbe3259b584d058c2e679427`;
- chunks SHA-256: `6b2ae2d4d87996d82b83c373c96cba78d3c0a10672b48990c71f35e54d8d1414`;
- prompt SHA-256: `905c77411aa7334bd74922a19d0cc708772e69d24ca5a421bd092e626cb41558`.

Nowy przebieg OpenAI zgadza się z dostępnymi plikami pytań i chunków.
Wcześniejszy problem zgodności z PR #72 nie dotyczy tego zapisu;
starszy wynik pozostał w historii Git.

Każda kombinacja była uruchomiona tylko raz. Mamy sześć lokalnych konfiguracji
i jedną w całości OpenAI, więc nie da się oddzielić wszystkich wpływów modeli.
Rankingi Qwen różnią się między smoke a przebiegami odpowiedzi: dla 4B w Q14
oraz Q01 z generatorem Qwen; dla 8B w Q10 oraz Q01 z Gemmą 31B.
Przyczyny nie ustalono. Generatory nie zawsze dostawały dokładnie ten sam kontekst.
Odpowiedzi wzorcowe mają nadal status `draft_not_page_verified`.

Alias 12B oznacza konfigurację 8K. Podsumowania nie zapisują wszystkich
ustawień kontekstu/KV-cache, digestów kwantyzacji, budżetów myślenia, liczby
tokenów wyjściowych ani stanu rozgrzania modeli. Czasy dotyczą tych uruchomień;
nie wystarczają do przypisania różnic samej architekturze.

Kolejność dalszej pracy: poprawić etykiety i ocenę przydatności fragmentów
(#75, #76), dodać diagnostykę błędów (#77), ustalić składnię cytowań (#78).
Do porównania szybkości warto zachować wektory pytań, ustalić identyczne
konteksty, rozgrzać jeden model odpowiedzi, zapisać ustawienia i powtórzyć pomiary.

## Dodatek: wszystkie nietrafienia dla pytań z odpowiedzią w korpusie

`present` oznacza, że oczekiwany dokument był w kontekście. Nie przesądza,
czy fragment zawierał odpowiedź. `absent` oznacza jego brak.

| Embeddings | Answer model | Question | Expected document | Observed outcome |
|---|---|---|---|---|
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q18 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | gemma4:12b-q8-8k | Q21 | absent | refusal |
| qwen3-embedding:4b | gemma4:31b | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | gemma4:31b | Q10 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | gemma4:31b | Q21 | absent | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q01 | absent | answered, no expected-document citation |
| qwen3-embedding:4b | qwen3.6:35b | Q10 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q21 | absent | refusal |
| qwen3-embedding:4b | qwen3.6:35b | Q22 | present (not proof of answer-bearing evidence) | answered, no expected-document citation; grouped-marker loss |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:12b-q8-8k | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | gemma4:31b | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q05 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q19 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | gemma4:31b | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | gemma4:31b | Q25 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | qwen3.6:35b | Q01 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | qwen3.6:35b | Q08 | unknown: error | ValidationError |
| qwen3-embedding:8b | qwen3.6:35b | Q10 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| qwen3-embedding:8b | qwen3.6:35b | Q21 | present (not proof of answer-bearing evidence) | refusal |
| qwen3-embedding:8b | qwen3.6:35b | Q25 | present (not proof of answer-bearing evidence) | refusal |
| text-embedding-3-small | gpt-5.4-mini | Q01 | absent | answered, no expected-document citation |
| text-embedding-3-small | gpt-5.4-mini | Q09 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
| text-embedding-3-small | gpt-5.4-mini | Q19 | present (not proof of answer-bearing evidence) | answered, no expected-document citation |
