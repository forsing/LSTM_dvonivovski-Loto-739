#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
1. Naučeni vektori brojeva.
2. Zajednički vremenski LSTM obrađuje istoriju svake pozicije.
3. Drugi LSTM povezuje opise sedam pozicija.
4. Regresija raspodela daje izlaz za sledeće izvlačenje.

Bez slučajne inicijalizacije, šuma, dropout-a, uzorkovanja
i mešanja podataka.

Treniranje koristi kvadratnu grešku kumulativnih raspodela,
a ne klasifikacionu funkciju greške.

Izbor kombinacije koristi dinamičko programiranje.
Nema zaokruživanja, pomeranja duplikata ni kozmetičkih filtera.
"""


import argparse
import csv
import hashlib
import math
import re
import sys
from pathlib import Path

try:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit(
        "Nedostaje biblioteka. Instalacija:\n"
        "python3 -m pip install numpy torch\n"
        f"Detalj: {exc}"
    )


DEFAULT_FILES = (
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_2974.csv",
    "/Users/4c/Desktop/GHQ/data/loto7_4704_k83_loto_plus_1730.csv",
)

BALLS = 39
PICKS = 7
WINDOW = 5

EMBEDDING_SIZE = 64
TEMPORAL_SIZE = 128
POSITION_SIZE = 64

VALIDATION_SIZE = 48
TEST_SIZE = 64
MIN_TRAIN_SAMPLES = 128

MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0008

DEVICE = torch.device("cpu")


def configure_torch():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def read_draws(filename):
    path = Path(filename).expanduser()

    if not path.is_file():
        raise ValueError(f"Fajl ne postoji: {path}")

    rows = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()

            if not line:
                continue

            if "," in line or ";" in line:
                delimiter = "," if "," in line else ";"
                fields = next(
                    csv.reader([line], delimiter=delimiter, strict=True)
                )
                fields = [field.strip() for field in fields]
            else:
                fields = re.split(r"\s+", line)

            if len(fields) != PICKS:
                raise ValueError(
                    f"Red {line_number}: očekivano 7 brojeva, "
                    f"pronađeno {len(fields)}."
                )

            if any(
                re.fullmatch(r"[0-9]+", field) is None
                for field in fields
            ):
                raise ValueError(
                    f"Red {line_number}: očekivani su celi brojevi "
                    "bez zaglavlja i dodatnih kolona."
                )

            numbers = [int(field) for field in fields]

            if any(number < 1 or number > BALLS for number in numbers):
                raise ValueError(
                    f"Red {line_number}: brojevi moraju biti 1–39."
                )

            if len(set(numbers)) != PICKS:
                raise ValueError(
                    f"Red {line_number}: broj se ponavlja."
                )

            # Hronologija redova ostaje nepromenjena.
            rows.append(sorted(numbers))

    required = (
        WINDOW
        + MIN_TRAIN_SAMPLES
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if len(rows) < required:
        raise ValueError(
            f"Potrebno najmanje {required} izvlačenja; "
            f"pronađeno {len(rows)}."
        )

    return path, np.asarray(rows, dtype=np.int64)


def prepare_data(draws):
    sample_count = len(draws) - WINDOW

    inputs = np.empty(
        (sample_count, WINDOW, PICKS),
        dtype=np.int64,
    )

    # Svako poznato izvlačenje posle početnog prozora
    # služi kao cilj jednog istorijskog prelaza.
    for sample in range(sample_count):
        inputs[sample] = draws[sample:sample + WINDOW] - 1

    outcomes = draws[WINDOW:].copy()

    grid = np.arange(1, BALLS + 1)
    target_cdf = (
        outcomes[:, :, None] <= grid[None, None, :]
    ).astype(np.float64)

    # NEXT prozor uključuje poslednje poznato izvlačenje.
    next_input = (draws[-WINDOW:] - 1)[None].copy()

    return inputs, target_cdf, outcomes, next_input


def analytic_fill(parameter, phase):
    """Analitička inicijalizacija, bez generatora slučajnih brojeva."""
    with torch.no_grad():
        if parameter.ndim == 1:
            parameter.zero_()
            return

        rows, columns = parameter.shape

        row = torch.arange(
            1,
            rows + 1,
            dtype=torch.float64,
            device=parameter.device,
        )[:, None]

        column = torch.arange(
            1,
            columns + 1,
            dtype=torch.float64,
            device=parameter.device,
        )[None, :]

        angles = (
            math.sqrt(2.0) * row * column
            + 0.113 * row.square()
            + 0.071 * column.square()
            + phase
        )

        values = torch.sin(angles) * math.sqrt(2.0 / columns)
        parameter.copy_(values.to(parameter.dtype))


class AnalyticEmbedding(nn.Embedding):
    def reset_parameters(self):
        analytic_fill(self.weight, phase=0.23)


class AnalyticLinear(nn.Linear):
    def reset_parameters(self):
        analytic_fill(self.weight, phase=2.71)

        if self.bias is not None:
            with torch.no_grad():
                self.bias.zero_()


class AnalyticLSTM(nn.LSTM):
    def __init__(self, input_size, hidden_size, phase):
        self.phase = float(phase)

        # Konstruktor koristi ovde definisan reset_parameters,
        # pa nema prethodne slučajne inicijalizacije.
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

    def reset_parameters(self):
        for index, (_, parameter) in enumerate(
            self.named_parameters(recurse=False)
        ):
            analytic_fill(
                parameter,
                self.phase + 0.379 * index,
            )

        with torch.no_grad():
            # Redosled kapija u PyTorch-u: i, f, g, o.
            self.bias_ih_l0[
                self.hidden_size:2 * self.hidden_size
            ].fill_(1.0)


class TwoLevelLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = AnalyticEmbedding(
            num_embeddings=BALLS,
            embedding_dim=EMBEDDING_SIZE,
        )

        # Isti vremenski LSTM primenjuje se na svaku poziciju.
        self.temporal_lstm = AnalyticLSTM(
            EMBEDDING_SIZE,
            TEMPORAL_SIZE,
            phase=0.89,
        )

        # Drugi LSTM obrađuje niz opisa sedam pozicija.
        self.position_lstm = AnalyticLSTM(
            TEMPORAL_SIZE,
            POSITION_SIZE,
            phase=1.67,
        )

        self.output = AnalyticLinear(POSITION_SIZE, BALLS)

        rank = torch.arange(PICKS)[:, None]
        number_index = torch.arange(BALLS)[None, :]

        # Strukturna podrška sortiranih pozicija:
        # 1. pozicija 1–33, ..., 7. pozicija 7–39.
        legal_support = (
            (number_index >= rank)
            & (number_index <= BALLS - PICKS + rank)
        )

        self.register_buffer("legal_support", legal_support)

    def forward(self, inputs):
        batch = inputs.shape[0]

        # (B, vreme, pozicija, vektor)
        embedded = self.embedding(inputs)

        # Svaka pozicija postaje zaseban vremenski niz.
        per_position = embedded.permute(
            0, 2, 1, 3
        ).contiguous().reshape(
            batch * PICKS,
            WINDOW,
            EMBEDDING_SIZE,
        )

        _, (hidden, _) = self.temporal_lstm(per_position)

        position_descriptions = hidden[-1].reshape(
            batch, PICKS, TEMPORAL_SIZE
        )

        connected, _ = self.position_lstm(position_descriptions)

        logits = self.output(connected).to(torch.float64)

        # Pozitivne mase, trenirane regresijom CDF.
        mass = F.softplus(logits) + 1e-12
        mass = mass * self.legal_support[None]

        return mass / mass.sum(dim=-1, keepdim=True)


def input_tensor(values):
    return torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.int64)
    ).to(DEVICE)


def target_tensor(values):
    return torch.from_numpy(
        np.ascontiguousarray(values, dtype=np.float64)
    ).to(DEVICE)


def regression_loss(mass, target_cdf):
    predicted_cdf = mass.cumsum(dim=-1)
    return (predicted_cdf - target_cdf).square().mean()


def train_epoch(model, optimizer, inputs, targets):
    model.train()

    total = 0.0
    count = 0

    for start in range(0, len(inputs), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(inputs))

        optimizer.zero_grad(set_to_none=True)

        mass = model(inputs[start:stop])
        loss = regression_loss(mass, targets[start:stop])

        if not torch.isfinite(loss):
            raise RuntimeError("Nekonačna greška pri treniranju.")

        loss.backward()

        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nekonačan gradijent.")

        optimizer.step()

        total += float(loss.detach()) * (stop - start)
        count += stop - start

    return total / count


def select_epochs(train_x, train_y, validation_x, validation_y):
    model = TwoLevelLSTM().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_inputs = input_tensor(train_x)
    train_targets = target_tensor(train_y)

    validation_inputs = input_tensor(validation_x)
    validation_targets = target_tensor(validation_y)

    best_epoch = 0
    best_error = float("inf")
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_error = train_epoch(
            model,
            optimizer,
            train_inputs,
            train_targets,
        )

        model.eval()

        with torch.no_grad():
            validation_error = float(
                regression_loss(
                    model(validation_inputs),
                    validation_targets,
                )
            )

        if not math.isfinite(validation_error):
            raise RuntimeError("Nekonačna validaciona greška.")

        if validation_error < best_error:
            best_error = validation_error
            best_epoch = epoch
            stale = 0
        else:
            stale += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  Izbor epoha {epoch}/{MAX_EPOCHS} | "
                f"trening={train_error:.6f} | "
                f"validacija={validation_error:.6f}",
                flush=True,
            )

        if stale >= PATIENCE:
            break

    if best_epoch < 1:
        raise RuntimeError("Izbor broja epoha nije uspeo.")

    return best_epoch, best_error


def fit_fixed(inputs, targets, epochs, label):
    model = TwoLevelLSTM().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    tensor_x = input_tensor(inputs)
    tensor_y = target_tensor(targets)

    for epoch in range(1, epochs + 1):
        error = train_epoch(
            model,
            optimizer,
            tensor_x,
            tensor_y,
        )

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"  {label}: {epoch}/{epochs} | "
                f"greška={error:.6f}",
                flush=True,
            )

    return model


def predict_masses(model, inputs):
    model.eval()
    blocks = []

    with torch.no_grad():
        for start in range(0, len(inputs), BATCH_SIZE):
            block = input_tensor(
                inputs[start:start + BATCH_SIZE]
            )
            blocks.append(model(block).cpu().numpy())

    masses = np.concatenate(blocks, axis=0)

    if not np.isfinite(masses).all():
        raise RuntimeError("Neispravan numerički izlaz modela.")

    return masses


def select_combination(mass):
    """
    Tačno maksimizuje proizvod naučenih masa sedam pozicija
    među svim strogo rastućim kombinacijama.

    Ograničenja su isključivo:
        1 <= n1 < n2 < ... < n7 <= 39.
    """
    if mass.shape != (PICKS, BALLS):
        raise ValueError("Pogrešan oblik raspodele.")

    if not np.isfinite(mass).all() or np.any(mass < 0):
        raise ValueError("Neispravne mase raspodele.")

    log_mass = np.full(
        (PICKS, BALLS), -np.inf, dtype=np.float64
    )
    np.log(mass, out=log_mass, where=mass > 0)

    scores = np.full(
        (PICKS, BALLS), -np.inf, dtype=np.float64
    )
    parents = np.full(
        (PICKS, BALLS), -1, dtype=np.int64
    )

    scores[0, :BALLS - PICKS + 1] = (
        log_mass[0, :BALLS - PICKS + 1]
    )

    for rank in range(1, PICKS):
        best_score = -np.inf
        best_index = -1

        for number_index in range(BALLS):
            previous = number_index - 1

            if (
                previous >= 0
                and scores[rank - 1, previous] > best_score
            ):
                best_score = scores[rank - 1, previous]
                best_index = previous

            if not (
                rank <= number_index <= BALLS - PICKS + rank
            ):
                continue

            if best_index >= 0:
                scores[rank, number_index] = (
                    best_score + log_mass[rank, number_index]
                )
                parents[rank, number_index] = best_index

    current = int(np.argmax(scores[-1]))

    if not np.isfinite(scores[-1, current]):
        raise RuntimeError("Nema važeće kombinacije.")

    result = [0] * PICKS
    result[-1] = current + 1

    for rank in range(PICKS - 1, 0, -1):
        current = int(parents[rank, current])

        if current < 0:
            raise RuntimeError("Neispravna putanja izbora.")

        result[rank - 1] = current + 1

    combination = tuple(result)

    if (
        len(set(combination)) != PICKS
        or combination[0] < 1
        or combination[-1] > BALLS
        or combination != tuple(sorted(combination))
    ):
        raise RuntimeError("Izlaz nije važeća Loto kombinacija.")

    return combination


def numbers_text(numbers):
    return ", ".join(str(int(number)) for number in numbers)


def longest_run(numbers):
    longest = current = 1

    for previous, following in zip(numbers[:-1], numbers[1:]):
        current = (
            current + 1 if following == previous + 1 else 1
        )
        longest = max(longest, current)

    return longest


def report_test(masses, targets, outcomes, reference_cdf):
    predicted_cdf = masses.cumsum(axis=-1)

    model_error = float(
        np.mean((predicted_cdf - targets) ** 2)
    )
    reference_error = float(
        np.mean((reference_cdf[None] - targets) ** 2)
    )

    hits = []

    for mass, actual in zip(masses, outcomes):
        prediction = select_combination(mass)
        hits.append(
            len(set(prediction).intersection(actual.tolist()))
        )

    counts = np.bincount(hits, minlength=PICKS + 1)

    print(f"  Provereno kola: {len(hits)}")
    print(f"  CDF regresiona greška: {model_error:.6f}")
    print(
        "  CDF greška istorijskog proseka, za poređenje: "
        f"{reference_error:.6f}"
    )
    print(f"  Prosečno pogodaka: {np.mean(hits):.4f}")
    print(
        "  Pogoci 0–7: "
        + " | ".join(
            f"{number}: {int(counts[number])}"
            for number in range(PICKS + 1)
        )
    )


def process_file(filename):
    path, draws = read_draws(filename)
    inputs, targets, outcomes, next_input = prepare_data(draws)

    train_end = (
        len(inputs) - VALIDATION_SIZE - TEST_SIZE
    )
    validation_end = len(inputs) - TEST_SIZE

    fingerprint = hashlib.sha256(
        draws.astype("<i8", copy=False).tobytes()
    ).hexdigest()[:16]

    print("\n" + "=" * 72)
    print(path.name)
    print(
        f"Izvlačenja: {len(draws)} | "
        f"istorijski prelazi: {len(inputs)} | "
        f"oznaka sadržaja: {fingerprint}",
        flush=True,
    )

    print("\n1. Hronološki izbor broja epoha")

    epochs, validation_error = select_epochs(
        inputs[:train_end],
        targets[:train_end],
        inputs[train_end:validation_end],
        targets[train_end:validation_end],
    )

    print(
        f"  Izabrano epoha: {epochs} | "
        f"validaciona greška: {validation_error:.6f}"
    )

    print("\n2. Završna provera na izdvojenim kolima")

    test_model = fit_fixed(
        inputs[:validation_end],
        targets[:validation_end],
        epochs,
        label="Model za proveru",
    )

    # Model je fiksiran pre testa.
    # Svaki test prozor sadrži samo tada poznata prethodna kola.
    test_masses = predict_masses(
        test_model,
        inputs[validation_end:],
    )

    reference_cdf = targets[:validation_end].mean(axis=0)

    report_test(
        test_masses,
        targets[validation_end:],
        outcomes[validation_end:],
        reference_cdf,
    )

    del test_model, test_masses

    print(
        "\n3. Završno treniranje na celoj poznatoj istoriji",
        flush=True,
    )

    final_model = fit_fixed(
        inputs,
        targets,
        epochs,
        label="Završni model",
    )

    next_mass = predict_masses(
        final_model,
        next_input,
    )[0]

    combination = select_combination(next_mass)

    print("\nKontrola: prethodno izvuceno kolo")
    print(numbers_text(draws[-1]))

    print("\nNEXT:")
    print(numbers_text(combination))

    low = sum(number <= 19 for number in combination)
    historical_copies = int(
        np.sum(
            np.all(
                draws == np.asarray(combination)[None],
                axis=1,
            )
        )
    )

    print(
        "\nOpis izlaza: "
        f"najduži uzastopni niz={longest_run(combination)}; "
        f"1–19: {low}; 20–39: {PICKS - low}; "
        f"ista kombinacija u istoriji: {historical_copies}."
    )
    print("Opis izlaza ne menja izabranu kombinaciju.")

    del final_model

    return path.name, combination


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Loto 7/39 — dvonivojski LSTM V1."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        help="Tri CSV putanje; bez argumenata koristi DEFAULT_FILES.",
    )

    args = parser.parse_args()

    if args.csv_files and len(args.csv_files) != 3:
        parser.error("Potrebno je proslediti tačno tri CSV putanje.")

    return args.csv_files or list(DEFAULT_FILES)


def main():
    filenames = parse_arguments()
    configure_torch()

    print("LOTO 7/39 — DVONIVOJSKI LSTM V1")
    print(
        "Naučeni vektori brojeva -> vremenski LSTM -> "
        "LSTM između pozicija -> regresija raspodela."
    )

    results = []
    failures = []

    for filename in filenames:
        try:
            results.append(process_file(filename))
        except (ValueError, OSError, RuntimeError, csv.Error) as exc:
            failures.append((str(filename), str(exc)))
            print(
                f"\nGREŠKA za {Path(filename).name}: {exc}",
                flush=True,
            )

    if results:
        print("\n" + "=" * 72)
        print("KONAČNI NEXT REZULTATI")

        for name, combination in results:
            print(f"{name}: ({numbers_text(combination)})")

        for index, (first_name, first) in enumerate(results):
            for second_name, second in results[index + 1:]:
                if first == second:
                    print(
                        f"\nIsti NEXT: {first_name} / {second_name}: "
                        f"({numbers_text(first)}). "
                        "Rezultat nije naknadno menjan."
                    )

    if failures:
        print(
            "\nZa neuspešno obrađene fajlove "
            "nema zamenskih kombinacija."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



"""
LOTO 7/39 — DVONIVOVSKI LSTM V1
Naučeni vektori brojeva -> vremenski LSTM -> LSTM između pozicija -> regresija raspodela.


========================================================================
loto7_4704_k83.csv
Izvlačenja: 4704 | istorijski prelazi: 4699 | oznaka sadržaja: 1badcf2632a9f966

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.124011 | validacija=0.105588
  Izbor epoha 10/80 | trening=0.073113 | validacija=0.069630
  Izbor epoha 20/80 | trening=0.072350 | validacija=0.070104
  Izabrano epoha: 13 | validaciona greška: 0.069466

2. Završna provera na izdvojenim kolima
  Model za proveru: 1/13 | greška=0.123825
  Model za proveru: 10/13 | greška=0.073050
  Model za proveru: 13/13 | greška=0.072828
  Provereno kola: 64
  CDF regresiona greška: 0.076993
  CDF greška istorijskog proseka, za poređenje: 0.073869
  Prosečno pogodaka: 1.3906
  Pogoci 0-7: 0: 17 | 1: 18 | 2: 18 | 3: 9 | 4: 2 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: 1/13 | greška=0.123572
  Završni model: 10/13 | greška=0.073099
  Završni model: 13/13 | greška=0.072901

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, 10, 13, 24, 27, 34, 39

Opis izlaza: najduži uzastopni niz=1; 1-19: 3; 20-39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.



========================================================================
loto7_4704_k83_loto_2974.csv
Izvlačenja: 2974 | istorijski prelazi: 2969 | oznaka sadržaja: 992bc400ed1af4f4

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.131250 | validacija=0.117627
  Izbor epoha 10/80 | trening=0.073293 | validacija=0.070952
  Izabrano epoha: 9 | validaciona greška: 0.070926

2. Završna provera na izdvojenim kolima
  Model za proveru: 1/9 | greška=0.131046
  Model za proveru: 9/9 | greška=0.073474
  Provereno kola: 64
  CDF regresiona greška: 0.072508
  CDF greška istorijskog proseka, za poređenje: 0.072467
  Prosečno pogodaka: 1.2500
  Pogoci 0-7: 0: 16 | 1: 22 | 2: 20 | 3: 6 | 4: 0 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: 1/9 | greška=0.130746
  Završni model: 9/9 | greška=0.073472

Kontrola: prethodno izvuceno kolo
9, 14, 17, 19, 24, 28, 35

NEXT:
1, 2, 13, 16, 27, 38, 39

Opis izlaza: najduži uzastopni niz=2; 1-19: 4; 20-39: 3; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.



========================================================================
loto7_4704_k83_loto_plus_1730.csv
Izvlačenja: 1730 | istorijski prelazi: 1725 | oznaka sadržaja: fc57041695a80b3c

1. Hronološki izbor broja epoha
  Izbor epoha 1/80 | trening=0.137705 | validacija=0.134043
  Izbor epoha 10/80 | trening=0.074454 | validacija=0.077628
  Izbor epoha 20/80 | trening=0.072552 | validacija=0.077921
  Izabrano epoha: 13 | validaciona greška: 0.077228

2. Završna provera na izdvojenim kolima
  Model za proveru: 1/13 | greška=0.137643
  Model za proveru: 10/13 | greška=0.074548
  Model za proveru: 13/13 | greška=0.073556
  Provereno kola: 64
  CDF regresiona greška: 0.073994
  CDF greška istorijskog proseka, za poređenje: 0.072636
  Prosečno pogodaka: 1.3906
  Pogoci 0–7: 0: 14 | 1: 18 | 2: 25 | 3: 7 | 4: 0 | 5: 0 | 6: 0 | 7: 0

3. Završno treniranje na celoj poznatoj istoriji
  Završni model: 1/13 | greška=0.137404
  Završni model: 10/13 | greška=0.074196
  Završni model: 13/13 | greška=0.073335

Kontrola: prethodno izvuceno kolo
7, 10, 13, 27, 28, 31, 38

NEXT:
1, 2, 13, 20, 27, 38, 39

Opis izlaza: najduži uzastopni niz=2; 1-19: 3; 20-39: 4; ista kombinacija u istoriji: 0.
Opis izlaza ne menja izabranu kombinaciju.


========================================================================
KONAČNI NEXT REZULTATI
loto7_4704_k83.csv: (1, 10, 13, 24, 27, 34, 39)
loto7_4704_k83_loto_2974.csv: (1, 2, 13, 16, 27, 38, 39)
loto7_4704_k83_loto_plus_1730.csv: (1, 2, 13, 20, 27, 38, 39)
"""



"""
LSTM u dva nivoa.
Prvi nivo prati svaku poziciju broja kroz prethodna izvlačenja. 
Drugi povezuje dobijene opise pozicija i uči odnose između njih. 

Sačuvati oba nivoa, a izlaz prilagodili regresiji raspodela za sedam brojeva, bez slučajnih koraka i naknadnog pomeranja duplikata.
"""
