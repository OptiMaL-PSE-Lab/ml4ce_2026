"""Cross-fitted hourly forecasts of the uncertain variables of the DER environment.

Forecasts are issued at every decision time, for the next ``FORECAST_HORIZON`` hours,
for the following variables:
- Outdoor temperature (°C)
- Solar irradiance (W/m²)
- PV generation (kW) of every prosumer
- Household load (kW) of every prosumer

**Note**: Tariffs are not forecast: they are published, and the environment reads them directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from torch import Tensor, nn
from tqdm import tqdm

from course_package.core import FLOW_LAG, FORECAST_HORIZON, FORECAST_STREAMS
from course_package.environment.constants import DEFAULT_LOCAL_TZ

WARMUP_HOURS = 24 * 10  # first hours skipped by metrics (baselines need history)
PROFILE_DAYS = 7  # days averaged by the recent-profile baseline


@dataclass
class ForecastData:
    """Hourly source series of the forecaster, aligned on one UTC index.

    Attributes
    ----------
    index : pd.DatetimeIndex
        Hourly UTC timestamps.
    external_temp, solar_ghi : NDArray
        Series of shape ``(T,)``.
    pv_generation, domestic_load : NDArray
        Series of shape ``(T, N)``, one column per prosumer.
    prosumer_ids : list[str]
        Prosumer identifiers, in column order.
    """

    index: pd.DatetimeIndex
    external_temp: NDArray
    solar_ghi: NDArray
    pv_generation: NDArray
    domestic_load: NDArray
    prosumer_ids: list[str]


class _GRUModel(nn.Module):
    """GRU model mapping an input window to a vector of forecasts."""

    def __init__(
        self, input_size: int, output_size: int, hidden_size: int, num_layers: int
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        """Return branch forecasts from the final recurrent state."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class RNNForecaster(nn.Module):
    """Parallel GRU forecaster for temperature, PV generation, load and irradiance vectors.

    Parameters
    ----------
    input_groups : dict[str, list[int]]
        Input column indices of the ``temp``, ``pv``, ``demand`` and ``ghi`` branches.
    n_prosumers : int
        Number of prosumers.
    horizon : int, default=FORECAST_HORIZON
        Number of time steps to forecast. The temperature branch forecasts ``horizon - 1``.
    hidden_size : int, default=64
        Number of hidden units per GRU branch.
    num_layers : int, default=1
        Number of recurrent layers per branch.
    """

    def __init__(
        self,
        input_groups: dict[str, list[int]],
        n_prosumers: int,
        horizon: int = FORECAST_HORIZON,
        hidden_size: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.input_groups = input_groups
        self.temp_net = _GRUModel(
            len(input_groups["temp"]), horizon - 1, hidden_size, num_layers
        )
        self.pv_net = _GRUModel(
            len(input_groups["pv"]), n_prosumers * horizon, hidden_size, num_layers
        )
        self.demand_net = _GRUModel(
            len(input_groups["demand"]), n_prosumers * horizon, hidden_size, num_layers
        )
        # Built last so the other branches keep their initialization.
        self.ghi_net = _GRUModel(
            len(input_groups["ghi"]), horizon, hidden_size, num_layers
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forecast each target group from windows ``(batch, window, features)``."""
        return (
            self.temp_net(x[..., self.input_groups["temp"]]),
            self.pv_net(x[..., self.input_groups["pv"]]),
            self.demand_net(x[..., self.input_groups["demand"]]),
            self.ghi_net(x[..., self.input_groups["ghi"]]),
        )


def read_forecast_file(
    path: str | Path,
    index: pd.DatetimeIndex,
    prosumer_ids: tuple[str, ...],
    horizon: int = FORECAST_HORIZON,
) -> tuple[dict[str, NDArray], dict]:
    """Load forecast vectors of the requested hours and prosumers.

    Parameters
    ----------
    path : str | Path
        Path to the ``.npz`` containing forecasts and metadata.
    index : pd.DatetimeIndex
        UTC hours of the environment; every hour must be in the file.
    prosumer_ids : tuple[str, ...]
        Prosumers of the environment, in order.
    horizon : int, default=FORECAST_HORIZON
        Expected number of leads.

    Returns
    -------
    tuple[dict[str, NDArray], dict]
        ``external_temp`` and ``solar_ghi`` of shape ``(T, H)``, ``pv_generation``
        and ``domestic_load`` of shape ``(N, T, H)``, and the file metadata.

    Raises
    ------
    ValueError
        If the file does not match the environment configuration.
    """
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Forecast file {path} does not exist. "
            "Generate it with run_forecaster_training.py."
        )
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(data["metadata_json"].item())
        file_index = pd.DatetimeIndex(pd.to_datetime(data["datetime"], utc=True))
        positions = file_index.get_indexer(index)
        if (positions < 0).any():
            raise ValueError(
                f"Forecast file {path} does not cover the environment's hours "
                f"({index[0]} .. {index[-1]}); it covers {file_index[0]} .. "
                f"{file_index[-1]}. Regenerate it with run_forecaster_training.py."
            )
        if metadata.get("horizon") != horizon:
            raise ValueError(
                f"Forecast file {path} has horizon {metadata.get('horizon')}, "
                f"expected {horizon}."
            )
        missing = sorted(set(FORECAST_STREAMS) - set(data.files))
        if missing:
            raise ValueError(
                f"Forecast file {path} has missing forecasts for: {missing}. "
                "Regenerate it with run_forecaster_training.py."
            )
        missing = sorted(set(prosumer_ids) - set(metadata["prosumer_ids"]))
        if missing:
            raise ValueError(f"Forecast file {path} has no rows for {missing}.")
        rows = [metadata["prosumer_ids"].index(pid) for pid in prosumer_ids]
        vectors = {}
        for stream in FORECAST_STREAMS:
            x = data[stream]
            x = x[positions] if x.ndim == 2 else x[rows][:, positions]
            vectors[stream] = np.asarray(x, np.float32)
    return vectors, metadata


def resolve_forecast_path(root_dir: str | Path, path: str | Path) -> Path:
    """Return the forecast file, relative to the dataset or the project root.

    Raises
    ------
        If the file is in neither place.
    """
    project = Path(__file__).resolve().parents[2]
    root_dir = Path(root_dir)
    if not root_dir.is_absolute():
        root_dir = project / root_dir
    path = Path(path)
    candidates = [path] if path.is_absolute() else [root_dir / path, project / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]  # will raise FileNotFoundError when opened


def check_forecast_leakage(
    env_config: dict, first_evaluation_time: pd.Timestamp
) -> None:
    """Raise if the configured forecast file was trained on evaluation data.

    Raises
    ------
    FileNotFoundError
        If the configured forecast file does not exist.
    ValueError
        If the file overlaps the evaluation block or lacks ``development_end``.
    """
    forecasting_data_path = env_config.get("data", {}).get("forecasting")
    if forecasting_data_path is None:
        return
    path = resolve_forecast_path(env_config["root_dir"], forecasting_data_path)
    with np.load(path, allow_pickle=False) as data:
        development_end = json.loads(data["metadata_json"].item()).get(
            "development_end"
        )
    first_evaluation_time = pd.Timestamp(first_evaluation_time)
    if development_end is None:
        raise ValueError(
            f"Forecast file {path} does not record its development_end; "
            "regenerate it with run_forecaster_training.py."
        )
    development_end = pd.Timestamp(development_end)
    if first_evaluation_time.tzinfo is None:
        first_evaluation_time = first_evaluation_time.tz_localize(development_end.tz)
    if development_end >= first_evaluation_time:
        raise ValueError(
            f"Forecast file {path} was trained on data up to {development_end}, "
            f"which overlaps the held-out block starting {first_evaluation_time}. "
            "Regenerate it with run_forecaster_training.py."
        )


def load_forecast_data(env_config: dict) -> ForecastData:
    """Read the source series of the forecaster from a loaded environment config.

    Parameters
    ----------
    env_config : dict
        Config as returned by ``load_config`` (with ``prosumers`` key).

    Returns
    -------
    ForecastData
        Series of the whole dataset.
    """
    root_dir = Path(env_config["root_dir"])
    if not root_dir.is_absolute():
        root_dir = Path(__file__).resolve().parents[2] / root_dir

    def read(file: str, columns: list[str]) -> pd.DataFrame:
        frame = pd.read_csv(root_dir / file, usecols=["datetime", *columns])
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        return frame.set_index("datetime")

    weather = read(env_config["data"]["weather"], ["external_temp", "solar_ghi"])
    prosumer_ids = list(env_config["prosumers"])
    pv, load = [], []
    for prosumer_id in prosumer_ids:
        frame = read(
            env_config["prosumers"][prosumer_id]["data_file"],
            ["pv_generation", "domestic_load"],
        )
        if not frame.index.equals(weather.index):
            raise ValueError(f"{prosumer_id} is not on the weather time index.")
        pv.append(frame["pv_generation"].to_numpy(np.float32))
        load.append(frame["domestic_load"].to_numpy(np.float32))
    if weather.isna().any().any() or np.isnan(pv).any() or np.isnan(load).any():
        raise ValueError("Forecast source series contain missing values.")
    return ForecastData(
        index=weather.index,
        external_temp=weather["external_temp"].to_numpy(np.float32),
        solar_ghi=weather["solar_ghi"].to_numpy(np.float32),
        pv_generation=np.stack(pv, axis=1),
        domestic_load=np.stack(load, axis=1),
        prosumer_ids=prosumer_ids,
    )


def lag_rows(values: NDArray, lag: int) -> NDArray:
    """Delay a series by ``lag`` rows: row ``t`` holds ``values[t - lag]``."""
    if lag == 0:
        return values
    return np.concatenate([np.repeat(values[:1], lag, axis=0), values[:-lag]])


def decision_features(
    data: ForecastData, flow_lag: int = FLOW_LAG, local_tz: str = DEFAULT_LOCAL_TZ
) -> tuple[NDArray, dict[str, list[int]]]:
    """Return what is known at each decision time (outdoor temperature at ``t``, the
    irradiance, PV generation and load up to ``t - flow_lag``), one row per hour.

    Returns
    -------
    tuple[NDArray, dict[str, list[int]]]
        Features of shape ``(T, F)`` and the input columns of each GRU branch.
    """
    local = data.index.tz_convert(local_tz)
    calendar = np.stack(
        [
            np.sin(2 * np.pi * local.hour / 24),
            np.cos(2 * np.pi * local.hour / 24),
            np.sin(2 * np.pi * local.dayofweek / 7),
            np.cos(2 * np.pi * local.dayofweek / 7),
            np.sin(2 * np.pi * local.month / 12),
            np.cos(2 * np.pi * local.month / 12),
        ],
        axis=1,
    )
    n_prosumers = len(data.prosumer_ids)
    features = np.concatenate(
        [
            data.external_temp[:, None],
            lag_rows(data.solar_ghi, flow_lag)[:, None],
            lag_rows(data.pv_generation, flow_lag),
            lag_rows(data.domestic_load, flow_lag),
            calendar,
        ],
        axis=1,
    ).astype(np.float32)
    pv = list(range(2, 2 + n_prosumers))
    load = list(range(2 + n_prosumers, 2 + 2 * n_prosumers))
    calendar_columns = list(range(2 + 2 * n_prosumers, features.shape[1]))
    return features, {
        "temp": [0, 1, *calendar_columns],
        "pv": [0, 1, *pv, *calendar_columns],
        "demand": [*load, *calendar_columns],
        "ghi": [0, 1, *calendar_columns],
    }


def lead_windows(values: NDArray, horizon: int) -> NDArray:
    """Return ``out[t, ..., k] = values[t + k]``, ``NaN`` beyond the series end."""
    padding = np.full((horizon - 1, *values.shape[1:]), np.nan, dtype=np.float32)
    return sliding_window_view(
        np.concatenate([values.astype(np.float32), padding]), horizon, axis=0
    )


def target_matrix(data: ForecastData, horizon: int) -> NDArray:
    """Return the training targets ``(T, (horizon - 1) + 2 * N * horizon + horizon)``.

    Columns: temperature leads ``1 .. horizon - 1``, then PV and load of every
    prosumer for leads ``0 .. horizon - 1`` (prosumer-major), then irradiance for
    leads ``0 .. horizon - 1``.
    """
    n_rows = len(data.index)
    temp = lead_windows(data.external_temp, horizon)[:, 1:]
    pv = lead_windows(data.pv_generation, horizon).reshape(n_rows, -1)
    load = lead_windows(data.domestic_load, horizon).reshape(n_rows, -1)
    ghi = lead_windows(data.solar_ghi, horizon)
    return np.concatenate([temp, pv, load, ghi], axis=1)


def split_outputs(
    outputs: NDArray, n_prosumers: int, horizon: int
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Split model outputs ``(R, n_out)`` into temperature, PV, load and irradiance groups."""
    n_flow = n_prosumers * horizon
    return (
        outputs[:, : horizon - 1],
        outputs[:, horizon - 1 : horizon - 1 + n_flow],
        outputs[:, horizon - 1 + n_flow : horizon - 1 + 2 * n_flow],
        outputs[:, horizon - 1 + 2 * n_flow :],
    )


def training_rows(
    n_rows: int,
    window: int,
    horizon: int,
    development_end_idx: int,
    held_out: tuple[int, int] | None = None,
    flow_lag: int = FLOW_LAG,
) -> NDArray:
    """Return the issue times a model may be trained on.

    A sample issued at ``t`` spans the hours ``t - window + 1 - flow_lag ..
    t + horizon - 1``: its inputs reach back ``flow_lag`` hours further for the
    lagged flows, and its targets reach forward ``horizon - 1`` hours. It must end
    inside the development period and, for a fold, be disjoint from the span of
    every sample of the held-out block.

    Parameters
    ----------
    n_rows : int
        Number of hours of the dataset.
    window : int
        Input window length.
    horizon : int
        Number of leads.
    development_end_idx : int
        Index of the last hour the targets may use.
    held_out : tuple[int, int] | None, default=None
        Issue times ``[lo, hi)`` of the block to leave out.
    flow_lag : int, default=FLOW_LAG
        Delay of the flow inputs.
    """
    t = np.arange(n_rows)
    start = t - window + 1 - flow_lag
    allowed = t + horizon - 1 <= development_end_idx
    if held_out is not None:
        lo, hi = held_out
        before = t + horizon - 1 < lo - window + 1 - flow_lag
        after = start > hi + horizon - 2
        allowed &= before | after
    return np.flatnonzero(allowed)


def _build_windows(features: NDArray, window: int) -> NDArray:
    """Return the input windows ``(T, window, F)`` ending at each decision time."""
    padded = np.concatenate([np.repeat(features[:1], window - 1, axis=0), features])
    return _sliding_windows(padded, window)


def _sliding_windows(data: NDArray, window_width: int) -> NDArray:
    """Return zero-copy overlapping windows ``(n_windows, window_width, n_features)``."""
    return np.moveaxis(sliding_window_view(data, window_width, axis=0), -1, 1)


@dataclass
class _FittedModel:
    """A trained forecaster with the scalers of its training rows."""

    model: RNNForecaster
    input_mean: Tensor
    input_std: Tensor
    target_mean: Tensor
    target_std: Tensor


def _fit(
    windows: Tensor,
    targets: Tensor,
    rows: NDArray,
    input_groups: dict[str, list[int]],
    n_prosumers: int,
    horizon: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[_FittedModel, list[float]]:
    """Train a forecaster on the given issue times and return its loss history."""
    device = windows.device
    index = torch.as_tensor(rows, device=device)
    train_windows = windows[index]
    train_targets = targets[index]
    input_mean = train_windows.mean(dim=(0, 1))
    input_std = train_windows.std(dim=(0, 1)).clamp_min(1e-6)
    target_mean = train_targets.mean(dim=0)
    target_std = train_targets.std(dim=0).clamp_min(1e-6)

    torch.manual_seed(seed)
    model = RNNForecaster(
        input_groups, n_prosumers, horizon, hidden_size, num_layers
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator(device=device).manual_seed(seed)

    history = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(index), device=device, generator=generator)
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            x = (train_windows[batch] - input_mean) / input_std
            y = (train_targets[batch] - target_mean) / target_std
            loss = sum(
                nn.functional.mse_loss(output, target)
                for output, target in zip(
                    model(x), split_outputs(y, n_prosumers, horizon), strict=True
                )
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(order))
    return _FittedModel(model, input_mean, input_std, target_mean, target_std), history


@torch.no_grad()
def _predict(
    fitted: _FittedModel, windows: Tensor, rows: NDArray, batch_size: int = 2048
) -> NDArray:
    """Return raw-scale model outputs for the given issue times."""
    fitted.model.eval()
    outputs = []
    for start in range(0, len(rows), batch_size):
        index = torch.as_tensor(rows[start : start + batch_size], device=windows.device)
        x = (windows[index] - fitted.input_mean) / fitted.input_std
        scaled = torch.cat(fitted.model(x), dim=1)
        outputs.append((scaled * fitted.target_std + fitted.target_mean).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def cross_fit(
    data: ForecastData,
    development_end_idx: int,
    n_folds: int = 4,
    window_width: int = 24,
    horizon: int = FORECAST_HORIZON,
    hidden_size: int = 64,
    num_layers: int = 1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 42,
    flow_lag: int = FLOW_LAG,
    local_tz: str = DEFAULT_LOCAL_TZ,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, NDArray], dict]:
    """Forecast every hour with a model that never saw the period it forecasts.

    Parameters
    ----------
    data : ForecastData
        Source series.
    development_end_idx : int
        Index of the last hour that may be used for training.
    n_folds : int, default=4
        Number of contiguous blocks of the development period.

    Returns
    -------
    tuple[dict[str, NDArray], dict]
        Forecast vectors (``external_temp`` and ``solar_ghi`` of shape ``(T, H)``,
        ``pv_generation`` and ``domestic_load`` of shape ``(N, T, H)``) and training info (fold
        edges and the training loss history of each model).
    """
    n_rows = len(data.index)
    n_prosumers = len(data.prosumer_ids)
    if not window_width <= development_end_idx + 1 <= n_rows:
        raise ValueError("development_end_idx is outside the dataset.")
    features, input_groups = decision_features(data, flow_lag, local_tz)
    windows = torch.as_tensor(
        _build_windows(features, window_width).copy(), device=device
    )
    targets = torch.as_tensor(target_matrix(data, horizon), device=device)

    n_dev = development_end_idx + 1
    edges = np.linspace(0, n_dev, n_folds + 1).astype(int)
    outputs = np.full((n_rows, targets.shape[1]), np.nan, dtype=np.float32)
    losses = {}

    def fit_and_predict(
        name: str, rows: NDArray, predict_rows: NDArray, fold_seed: int
    ):
        fitted, history = _fit(
            windows,
            targets,
            rows,
            input_groups,
            n_prosumers,
            horizon,
            hidden_size,
            num_layers,
            epochs,
            batch_size,
            lr,
            fold_seed,
        )
        outputs[predict_rows] = _predict(fitted, windows, predict_rows)
        losses[name] = history

    for fold in tqdm(range(n_folds), desc="Cross-fitting"):
        lo, hi = int(edges[fold]), int(edges[fold + 1])
        rows = training_rows(
            n_rows,
            window_width,
            horizon,
            development_end_idx,
            held_out=(lo, hi),
            flow_lag=flow_lag,
        )
        fit_and_predict(f"fold_{fold}", rows, np.arange(lo, hi), seed + fold)
    if n_dev < n_rows:
        rows = training_rows(
            n_rows, window_width, horizon, development_end_idx, flow_lag=flow_lag
        )
        fit_and_predict("final", rows, np.arange(n_dev, n_rows), seed + n_folds)

    temp, pv, load, ghi = split_outputs(outputs, n_prosumers, horizon)
    vectors = {
        "external_temp": np.concatenate([data.external_temp[:, None], temp], axis=1),
        "solar_ghi": np.clip(ghi, 0.0, None),
        "pv_generation": np.clip(
            pv.reshape(n_rows, n_prosumers, horizon).transpose(1, 0, 2), 0.0, None
        ),
        "domestic_load": np.clip(
            load.reshape(n_rows, n_prosumers, horizon).transpose(1, 0, 2), 0.0, None
        ),
    }
    vectors = {name: array.astype(np.float32) for name, array in vectors.items()}
    return vectors, {"fold_edges": edges.tolist(), "loss_history": losses}


def baseline_vectors(
    data: ForecastData, flow_lag: int, horizon: int, rows: NDArray
) -> dict[str, dict[str, NDArray]]:
    """Return causal reference forecasts for the issue times ``rows``.

    ``persistence`` repeats the latest known value. ``recent_profile`` averages
    the same hour of the last ``PROFILE_DAYS`` days that are already known at the
    decision time.
    """
    lead = np.arange(horizon)
    series = {
        "external_temp": (data.external_temp[:, None], 0),
        "solar_ghi": (data.solar_ghi[:, None], flow_lag),
        "pv_generation": (data.pv_generation, flow_lag),
        "domestic_load": (data.domestic_load, flow_lag),
    }
    out: dict[str, dict[str, NDArray]] = {}
    for stream, (values, lag) in series.items():
        persistence = lag_rows(values, lag)[rows]  # (R, N)
        first_day = np.maximum(-(-(lead + lag) // 24), 1)  # ceil((k + lag) / 24)
        profile = np.empty((len(rows), horizon, values.shape[1]), dtype=np.float32)
        for start in range(0, len(rows), 4000):
            chunk = rows[start : start + 4000]
            days = first_day[None, :, None] + np.arange(PROFILE_DAYS)[None, None, :]
            source = np.clip(
                chunk[:, None, None] + lead[None, :, None] - 24 * days, 0, None
            )
            profile[start : start + 4000] = values[source].mean(axis=2)
        out[stream] = {
            "persistence": np.repeat(persistence[:, None, :], horizon, axis=1),
            "recent_profile": profile,
        }
    return out


def lead_rmse(prediction: NDArray, truth: NDArray) -> list[float]:
    """Root mean squared error per lead; both arrays are ``(R, H, N)``."""
    error = np.mean((prediction - truth) ** 2, axis=(0, 2))
    return np.sqrt(error).astype(float).tolist()


def evaluate_forecasts(
    data: ForecastData,
    vectors: dict[str, NDArray],
    rows: NDArray,
    flow_lag: int = FLOW_LAG,
    horizon: int = FORECAST_HORIZON,
) -> dict[str, dict[str, list[float]]]:
    """Return the RMSE per lead of the model and its baselines on issue times ``rows``.

    Only rows whose every lead has an observed truth are scored.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        ``metrics[stream][method]`` is the RMSE of each lead ``0 .. horizon - 1``.
    """
    rows = rows[(rows >= WARMUP_HOURS) & (rows + horizon - 1 < len(data.index))]
    baselines = baseline_vectors(data, flow_lag, horizon, rows)
    truths = {
        "external_temp": data.external_temp[:, None],
        "solar_ghi": data.solar_ghi[:, None],
        "pv_generation": data.pv_generation,
        "domestic_load": data.domestic_load,
    }
    metrics: dict[str, dict[str, list[float]]] = {}
    for stream, truth_series in truths.items():
        truth = lead_windows(truth_series, horizon)[rows].transpose(0, 2, 1)
        if vectors[stream].ndim == 2:
            model = vectors[stream][rows][:, :, None]
        else:
            model = vectors[stream][:, rows].transpose(1, 2, 0)
        metrics[stream] = {"model": lead_rmse(model, truth)}
        for method, prediction in baselines[stream].items():
            metrics[stream][method] = lead_rmse(prediction, truth)
    return metrics


def plot_learning_curves(
    loss_history: dict[str, list[float]], output: str | Path
) -> None:
    """Plot the training loss of every cross-fitting model."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for name, history in loss_history.items():
        ax.plot(history, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (standardized MSE)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def save_forecasts(
    output: str | Path,
    index: pd.DatetimeIndex,
    vectors: dict[str, NDArray],
    metadata: dict[str, object],
) -> None:
    """Write forecast vectors and their metadata for the environment."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        datetime=np.asarray(index.astype(str), dtype="U32"),
        external_temp=vectors["external_temp"],
        solar_ghi=vectors["solar_ghi"],
        pv_generation=vectors["pv_generation"],
        domestic_load=vectors["domestic_load"],
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def train_forecaster(
    env_config: dict,
    output_path: str | Path,
    development_end: str | pd.Timestamp,
    learning_curve_path: str | Path | None = None,
    device: str | None = None,
    **fit_kwargs,
) -> dict:
    """Cross-fit the forecaster and write the forecast file.

    Parameters
    ----------
    env_config : dict
        Loaded environment config (see ``load_forecast_data``).
    output_path : str | Path
        Destination ``.npz`` file.
    development_end : str | pd.Timestamp
        Last hour that may be used for training; later hours are the test period.
        Use the end of the development block of ``split_env_config_by_episode``.
    learning_curve_path : str | Path | None, default=None
        Optional figure of the training losses.
    device : str | None, default=None
        Torch device; CUDA when available, otherwise CPU.
    **fit_kwargs
        Options of ``cross_fit`` (``n_folds``, ``epochs``, ``hidden_size``, ...).

    Returns
    -------
    dict
        Metadata written to the file, including per-lead RMSE of the model and its
        baselines on the cross-fitted development period and on the test period.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = load_forecast_data(env_config)
    development_end = pd.Timestamp(development_end)
    if development_end.tzinfo is None:
        development_end = development_end.tz_localize("UTC")
    development_end_idx = (
        int(data.index.searchsorted(development_end, side="right")) - 1
    )
    local_tz = env_config["scenarios"].get("local_tz", DEFAULT_LOCAL_TZ)
    horizon = fit_kwargs.get("horizon", FORECAST_HORIZON)
    flow_lag = fit_kwargs.get("flow_lag", FLOW_LAG)

    vectors, info = cross_fit(
        data, development_end_idx, device=device, local_tz=local_tz, **fit_kwargs
    )
    all_rows = np.arange(len(data.index))
    development_rows = all_rows[all_rows + horizon - 1 <= development_end_idx]
    test_rows = all_rows[all_rows > development_end_idx]
    metrics = {
        "cross_fitted_development": evaluate_forecasts(
            data, vectors, development_rows, flow_lag, horizon
        ),
        "test": evaluate_forecasts(data, vectors, test_rows, flow_lag, horizon),
    }
    metadata = {
        "model": "cross-fitted GRU",
        "horizon": horizon,
        "flow_lag": flow_lag,
        "prosumer_ids": data.prosumer_ids,
        "development_end": str(data.index[development_end_idx]),
        "data_end": str(data.index[-1]),
        "fold_edges": info["fold_edges"],
        "options": {
            key: value for key, value in fit_kwargs.items() if key not in {"device"}
        },
        "metrics": metrics,
    }
    save_forecasts(output_path, data.index, vectors, metadata)
    if learning_curve_path is not None:
        plot_learning_curves(info["loss_history"], learning_curve_path)
    return metadata
