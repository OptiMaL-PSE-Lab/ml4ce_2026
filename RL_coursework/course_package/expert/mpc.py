"""MPC of the EV battery of one prosumer: smart charging and V2G.

At every hour the controller solves, over ``k = 0 .. N - 1`` hours ahead (steps
of one hour, energies in kWh)::

    min  sum_k  p_imp[k] g_imp[k] - p_exp[k] g_exp[k]         grid cost of the household
              + k_deg (b_in[k] + b_out[k])                   battery degradation
         + k_unmet  sum_s eps[s]                             kWh missing from the target
         + k_buf    sum_s eps_buf[s]                         kWh missing from the buffer
         - p_T SoC[N]                                        value of the energy left

    s.t. SoC[0] = SoC0,  SoC[k + 1] = SoC[k] + b_in[k] - b_out[k]
         g_imp[k] - g_exp[k] = net[k] + b_in[k] / eta_ch - eta_dis b_out[k]
         net[k] = domestic_load[k] - PV[k] + heat_pump[k]
         b_in[k] <= P_in mu[k] y[k],   b_out[k] <= P_out mu[k] (1 - y[k])
         g_imp[k] <= G z[k],           g_exp[k] <= G (1 - z[k])
         SoC_min <= SoC[s] <= SoC_max
         SoC[s] + eps[s] + eps_buf[s] >= E_req[s] (1 + f),   eps_buf[s] <= f E_req[s]

``E_req[s]`` is the declared target at the departure ``s = d`` (``s = 0 .. N``) and
zero elsewhere, and ``f`` the buffer fraction of ``BatteryCommand`` (0 by default).
Since ``k_buf < k_unmet``, ``eps`` is the shortfall below the target and
``eps_buf`` the one inside the buffer. The costs are the environment's
(``CostParameters``), so with ``f = 0`` the objective is the environment's cost
of the horizon minus the linear terminal value ``price_saved_energy * SoC[N]``.
``y`` and ``z`` are binary only in the hours when simultaneous flows could pay
(an export price above the import price or a negative price); in the other hours
they are relaxed, so most problems are LPs. The problem is passed to HiGHS
(``highspy``) as sparse arrays at every solve: ``SoC[0]`` is fixed through its
bounds and the buffer band is the upper bound of ``eps_buf``.

Both information modes solve the same problem and differ only in their inputs:
``forecast`` uses what the environment publishes at decision time (exact prices
of the published window, then their forecast, and the load and PV forecasts),
``perfect`` the realized future prices, load and PV. The EV's departure time and
target always come from the user's declaration.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import highspy
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_matrix

from course_package.core.cost import CostParameters
from course_package.core.params import EVParams
from course_package.environment.forecasting import FORECAST_HORIZON

if TYPE_CHECKING:
    from course_package.environment import DERMultiAgentEnv

Information = Literal["perfect", "forecast"]


@dataclass(frozen=True)
class BatteryCommand:
    """Per-step modifications of the MPC set by a supervisory policy.

    The defaults leave the MPC unchanged.

    Attributes
    ----------
    terminal_price : float | None, default=None
        Value per kWh of the energy left at the end of the horizon. ``None`` uses
        ``MPCParameters.default_terminal_price``.
    terminal_price_scale : float, default=1.0
        Multiplier of the terminal price.
    buffer_fraction : float, default=0.0
        Reserve above the departure target, as a fraction of the target. A kWh
        missing from the reserve costs ``CostParameters.k_ev_buffer``.
    """

    terminal_price: float | None = None
    terminal_price_scale: float = 1.0
    buffer_fraction: float = 0.0


@dataclass
class MPCParameters:
    """Battery and economic parameters of the MPC of one prosumer.

    Attributes
    ----------
    N : int
        Horizon in hours.
    ev_param : EVParams
        Parameters characterizing the EV battery of a prosumer (scalars, see
        ``HouseParams.agent``). V2G is disabled when ``ev_b_out_max`` is 0.
    cost_param : CostParameters
        Parameters to compute the cost and related KPIs: ``k_battery_degradation``
        prices the throughput, ``k_ev_unmet`` a kWh missing from the departure
        target, ``k_ev_buffer`` a kWh missing from the reserve above it and
        ``price_saved_energy`` (if positive) the energy left at the end of the
        horizon.

    Raises
    ------
    ValueError
        If ``k_ev_buffer`` exceeds ``k_ev_unmet``: the reserve must be the cheaper
        shortfall, or ``eps`` would no longer measure the missing target.
    """

    N: int
    ev_param: "EVParams"
    cost_param: "CostParameters"

    def __post_init__(self):
        if self.cost_param.k_ev_buffer > self.cost_param.k_ev_unmet:
            raise ValueError("k_ev_buffer must not exceed k_ev_unmet.")

    def mobility(
        self, env: "DERMultiAgentEnv", agent_idx: int = 0
    ) -> tuple[NDArray, NDArray]:
        """Return availability ``(N,)`` and the departure requirement ``(N + 1,)``.
        Car is available (mu=1) is 1 while the car is plugged in before a declared departure.
        Problem ends at departure; trips and the next arrival are not modelled.
        """
        availability = np.zeros(self.N, dtype=np.float32)
        requirement = np.zeros(self.N + 1, dtype=np.float32)
        idx = min(env.current_step_global, env.ev_availability.shape[-1] - 1)
        if env.ev_availability[agent_idx, idx] <= 0.0:
            return availability, requirement
        departure = max(0, round(env.ev_time_to_departure[agent_idx, idx]))
        availability[: min(departure, self.N)] = 1.0
        if departure <= self.N:
            target = float(env.ev_required_soc_departure[agent_idx, idx])
            requirement[departure] = min(
                target, float(self.ev_param.ev_battery_capacity)
            )
        return availability, requirement

    def prices(
        self, env: "DERMultiAgentEnv", information: Information = "forecast"
    ) -> tuple[NDArray, NDArray]:
        """Return the import and export prices of the horizon."""
        if information == "forecast":
            self._check_forecast_horizon(env)
            return (
                env.forecast_vector("import_cost")[: self.N],
                env.forecast_vector("export_price")[: self.N],
            )
        idx = self._future_indices(env, env.import_cost.shape[-1])
        return env.import_cost[idx], env.export_price[idx]

    def net_load(
        self,
        env: "DERMultiAgentEnv",
        information: Information = "forecast",
        agent_idx: int = 0,
    ) -> NDArray:
        """Return the household's net import without the battery over the horizon.

        Load minus PV, plus the heat pump's electric power of the last step: the
        battery is planned around the thermal controller, whose future actions it
        does not know.
        """
        if information == "forecast":
            self._check_forecast_horizon(env)
            load = env.forecast_vector("domestic_load", agent_idx)[: self.N]
            pv = env.forecast_vector("pv_generation", agent_idx)[: self.N]
        else:
            idx = self._future_indices(env, env.domestic_load.shape[-1])
            load = env.domestic_load[agent_idx, idx]
            pv = env.pv_generation[agent_idx, idx]
        heat_pump = 0.0
        if env.current_step > 0:
            step = env.current_step - 1
            thermal = (
                env.heating_load[agent_idx, step] + env.cooling_load[agent_idx, step]
            )
            heat_pump = float(thermal) / float(env.param.thermal.hp_cop[agent_idx])
        return (load - pv + heat_pump).astype(np.float32)

    def default_terminal_price(self, import_price: NDArray) -> float:
        """Return the value per kWh of the energy left at the end of the horizon.

        ``cost_param.price_saved_energy`` if positive, otherwise the cost of
        recharging the energy at the cheapest import price of the horizon.
        """
        if self.cost_param.price_saved_energy > 0.0:
            return float(self.cost_param.price_saved_energy)
        return max(float(np.min(import_price)), 0.0) / float(
            self.ev_param.ev_ch_efficiency
        )

    def SoC0(self, env: "DERMultiAgentEnv", agent_idx: int = 0) -> float:
        """Return the current state of charge, within the battery limits."""
        soc = float(env.ev_soc[agent_idx, env.current_step])
        soc_min = float(self.ev_param.ev_soc_min)
        return min(max(soc, soc_min), float(self.ev_param.ev_battery_capacity))

    def _future_indices(self, env: "DERMultiAgentEnv", length: int) -> NDArray:
        """Return the indices of the horizon, clipped to the end of the data."""
        return np.minimum(env.current_step_global + np.arange(self.N), length - 1)

    def _check_forecast_horizon(self, env: "DERMultiAgentEnv") -> None:
        if self.N > env.forecast_horizon:
            raise ValueError(
                f"MPC horizon {self.N} exceeds the environment's forecast horizon "
                f"{env.forecast_horizon}."
            )


@dataclass(frozen=True)
class MPCPlan:
    """Optimal plan of one solve over the horizon (energies in kWh).

    Attributes
    ----------
    b_in, b_out : NDArray
        Charging and discharging energy of each hour ``(N,)``.
    g_imp, g_exp : NDArray
        Grid import and export of each hour ``(N,)``.
    SoC : NDArray
        State of charge at ``s = 0 .. N`` ``(N + 1,)``.
    eps, eps_buf : NDArray
        Shortfall below the departure target and inside the buffer ``(N + 1,)``.
    objective : float
        Optimal objective value.
    """

    b_in: NDArray
    b_out: NDArray
    g_imp: NDArray
    g_exp: NDArray
    SoC: NDArray
    eps: NDArray
    eps_buf: NDArray
    objective: float


@dataclass(frozen=True)
class _Problem:
    """Per-step inputs of the HiGHS problem, in ``Highs.passModel`` layout."""

    col_cost: NDArray
    col_lower: NDArray
    col_upper: NDArray
    row_lower: NDArray
    row_upper: NDArray
    matrix: csc_matrix
    integrality: NDArray
    plugged_in: bool


class BatteryMPC:
    """MPC controller returning the normalized battery action of one prosumer.

    Parameters
    ----------
    param : MPCParameters
        MPC parameters.
    information : Literal["perfect", "forecast"], default="forecast"
        Information mode.

    Attributes
    ----------
    plan : MPCPlan | None
        Plan of the last optimal solve, ``None`` before the first one.
    """

    VARIABLES = ("b_in", "b_out", "g_imp", "g_exp", "y", "z", "SoC", "eps", "eps_buf")
    """Column blocks of the problem, in order; the last three have ``N + 1`` entries."""

    def __init__(
        self, param: MPCParameters, information: Information = "forecast"
    ) -> None:
        self.param = param
        self.N = param.N
        self.information = information
        sizes = [self.N] * 6 + [self.N + 1] * 3
        starts = np.concatenate([[0], np.cumsum(sizes)])
        self._cols = {
            name: np.arange(start, start + size)
            for name, start, size in zip(self.VARIABLES, starts, sizes)
        }
        self._num_col = int(starts[-1])
        self._problem: _Problem | None = None
        self.plan: MPCPlan | None = None
        self.solver = highspy.Highs()
        self.solver.setOptionValue("output_flag", False)

    def __call__(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: BatteryCommand | None = None,
    ) -> float:
        """Solve the MPC and return the normalized battery action."""
        return self.solve(env, agent_idx, command)[-1]

    def update(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: BatteryCommand | None = None,
    ) -> None:
        """Set the problem inputs from the environment at its current step.

        Parameters
        ----------
        env : DERMultiAgentEnv
            Environment at the decision step.
        agent_idx : int, default=0
            Prosumer index.
        command : BatteryCommand | None, default=None
            Terminal price and SoC buffer of this solve only; ``None`` uses the
            parameters.

        Note
        ----
        The MPC relaxes y and z to continuous variables in [0, 1] in the hours
        when simultaneous flows cannot improve the objective (that option would not
        be optimal) in order to speed up the optimization (solution of a LP is
        cheaper than a MILP).
        # TODO: revise the elements in the battery command
        """
        param, N, cols = self.param, self.N, self._cols
        ev, cost_param = param.ev_param, param.cost_param
        p_in, p_out = float(ev.ev_b_in_max), float(ev.ev_b_out_max)
        eta_ch, eta_dis = float(ev.ev_ch_efficiency), float(ev.ev_disch_efficiency)
        availability, requirement = param.mobility(env, agent_idx)
        import_price, export_price = param.prices(env, self.information)
        net_load = param.net_load(env, self.information, agent_idx)
        mu = availability.astype(np.float64)
        requirement = requirement.astype(np.float64)

        command = BatteryCommand() if command is None else command
        terminal_price = command.terminal_price
        if terminal_price is None:
            terminal_price = param.default_terminal_price(import_price)
        terminal_price = float(terminal_price * command.terminal_price_scale)
        fraction = float(command.buffer_fraction)
        grid_bound = (
            float(np.max(np.abs(net_load))) + p_in / eta_ch + p_out * eta_dis + 1.0
        )

        cost = np.zeros(self._num_col)
        cost[cols["b_in"]] = cost[cols["b_out"]] = cost_param.k_battery_degradation
        cost[cols["g_imp"]] = import_price
        cost[cols["g_exp"]] = -export_price
        cost[cols["eps"]] = cost_param.k_ev_unmet
        cost[cols["eps_buf"]] = cost_param.k_ev_buffer
        cost[cols["SoC"][-1]] = -terminal_price

        lower = np.zeros(self._num_col)
        upper = np.full(self._num_col, highspy.kHighsInf)
        upper[cols["y"]] = upper[cols["z"]] = 1.0
        lower[cols["SoC"]] = float(ev.ev_soc_min)
        upper[cols["SoC"]] = float(ev.ev_battery_capacity)
        lower[cols["SoC"][0]] = upper[cols["SoC"][0]] = param.SoC0(env, agent_idx)
        upper[cols["eps_buf"]] = requirement * fraction

        # Rows, in blocks of N: dynamics, balance, charge, discharge, import,
        # export limits; then N + 1 mobility rows.
        t, s, ones = np.arange(N), np.arange(N + 1), np.ones(N)
        block = [(k * N + t) for k in range(6)]
        mobility = 6 * N + s
        entries = [
            (block[0], cols["SoC"][1:], ones),
            (block[0], cols["SoC"][:-1], -ones),
            (block[0], cols["b_in"], -ones),
            (block[0], cols["b_out"], ones),
            (block[1], cols["g_imp"], ones),
            (block[1], cols["g_exp"], -ones),
            (block[1], cols["b_in"], -ones / eta_ch),
            (block[1], cols["b_out"], ones * eta_dis),
            (block[2], cols["b_in"], ones),
            (block[2], cols["y"], -p_in * mu),
            (block[3], cols["b_out"], ones),
            (block[3], cols["y"], p_out * mu),
            (block[4], cols["g_imp"], ones),
            (block[4], cols["z"], -grid_bound * ones),
            (block[5], cols["g_exp"], ones),
            (block[5], cols["z"], grid_bound * ones),
            (mobility, cols["SoC"], np.ones(N + 1)),
            (mobility, cols["eps"], np.ones(N + 1)),
            (mobility, cols["eps_buf"], np.ones(N + 1)),
        ]
        rows, columns, values = (np.concatenate(x) for x in zip(*entries))
        nonzero = values != 0.0
        num_row = 6 * N + N + 1
        matrix = csc_matrix(
            (values[nonzero], (rows[nonzero], columns[nonzero])),
            shape=(num_row, self._num_col),
        )
        matrix.sort_indices()

        inf = highspy.kHighsInf
        row_lower = np.concatenate([
            np.zeros(N),
            net_load,
            np.full(4 * N, -inf),
            requirement * (1.0 + fraction),
        ])
        row_upper = np.concatenate([
            np.zeros(N),
            net_load,
            np.zeros(N),
            p_out * mu,
            np.zeros(N),
            np.full(N, grid_bound),
            np.full(N + 1, inf),
        ])

        needs_binaries = (import_price < export_price) | (
            np.minimum(import_price, export_price) < 0.0
        )
        integrality = np.zeros(self._num_col, dtype=np.int32)
        integrality[cols["y"]] = integrality[cols["z"]] = needs_binaries.astype(
            np.int32
        )
        self._problem = _Problem(
            col_cost=cost,
            col_lower=lower,
            col_upper=upper,
            row_lower=row_lower.astype(np.float64),
            row_upper=row_upper.astype(np.float64),
            matrix=matrix,
            integrality=integrality,
            plugged_in=bool(mu[0] > 0.0),
        )

    def solve(
        self,
        env: "DERMultiAgentEnv",
        agent_idx: int = 0,
        command: BatteryCommand | None = None,
    ) -> tuple[float, float, float]:
        """Solve the MPC at the current step.

        ``command`` modifies this solve only (see ``update``).

        Returns
        -------
        tuple[float, float, float]
            First-hour charging and discharging energy, and the normalized action.
        """
        self.update(env, agent_idx, command)
        problem = self._problem
        if not problem.plugged_in:
            return 0.0, 0.0, 0.0

        matrix = problem.matrix
        self.solver.passModel(
            self._num_col,
            matrix.shape[0],
            matrix.nnz,
            int(highspy.MatrixFormat.kColwise),
            int(highspy.ObjSense.kMinimize),
            0.0,
            problem.col_cost,
            problem.col_lower,
            problem.col_upper,
            problem.row_lower,
            problem.row_upper,
            matrix.indptr.astype(np.int32),
            matrix.indices.astype(np.int32),
            matrix.data,
            problem.integrality,
        )
        self.solver.run()
        if self.solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
            warnings.warn(
                "Battery MPC found no optimal solution; holding the battery idle.",
                stacklevel=2,
            )
            return 0.0, 0.0, 0.0
        x = np.asarray(self.solver.getSolution().col_value)
        self.plan = MPCPlan(
            **{
                name: x[self._cols[name]]
                for name in ("b_in", "b_out", "g_imp", "g_exp", "SoC", "eps", "eps_buf")
            },
            objective=float(self.solver.getInfo().objective_function_value),
        )
        b_in, b_out = float(self.plan.b_in[0]), float(self.plan.b_out[0])
        return b_in, b_out, self._normalized_action(b_in, b_out)

    def _normalized_action(self, b_in: float, b_out: float) -> float:
        """Map first-hour energies to the battery action: energy / rated power."""
        ev = self.param.ev_param
        if b_in >= b_out:
            return float(np.clip(b_in / float(ev.ev_b_in_max), 0.0, 1.0))
        return float(np.clip(-b_out / float(ev.ev_b_out_max), -1.0, 0.0))


def initialize_mpc_param(
    env: "DERMultiAgentEnv", agent_idx: int = 0, N: int = FORECAST_HORIZON
) -> MPCParameters:
    """Initialize MPC parameters of one prosumer from the environment.

    Parameters
    ----------
    env : DERMultiAgentEnv
        Environment containing the prosumer's battery and heat-pump parameters.
    agent_idx : int, default=0
        Index of the prosumer.
    N : int, default=FORECAST_HORIZON
        Horizon in hours.

    Returns
    -------
    MPCParameters
        Configured MPC parameters.
    """
    house_param = env.param.agent(agent_idx)
    return MPCParameters(N=N, ev_param=house_param.ev, cost_param=env.cost_param)
