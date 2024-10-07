"""Functions for simulations based on pyvmte."""

# from pyvmte.utilities import

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pyvmte.classes import Estimand, Instrument  # type: ignore[import-untyped]
from pyvmte.estimation import estimation  # type: ignore[import-untyped]
from pyvmte.identification import identification  # type: ignore[import-untyped]
from pyvmte.utilities import (  # type: ignore[import-untyped]
    generate_bernstein_basis_funcs,
    generate_constant_splines_basis_funcs,
    mtr_funcs_from_solution,
)

from thesis.classes import LocalATEs
from thesis.config import RNG
from thesis.utilities import make_mtr_binary_iv, simulate_data_from_mtrs_binary_iv


def simulation_pyvmte(
    num_sims: int,
    num_obs: int,
    instrument: Instrument,
    idestimands: str,
    bfunc_type: str,
    bfunc_options: dict,
    constraints: dict,
    local_ates: LocalATEs,
    confidence_interval: str,
    confidence_interval_options: dict,
    u_hi_extra: float = 0.2,
) -> pd.DataFrame:
    """Perform simulation on the binary IV model using the pyvmte package."""
    # ----------------------------------------------------------------------------------
    # Preliminary computations
    # ----------------------------------------------------------------------------------
    pscore_lo = instrument.pscores[0]
    pscore_hi = instrument.pscores[1]

    target_for_id = Estimand(
        "late",
        u_lo=pscore_lo,
        u_hi=pscore_hi,
        u_hi_extra=u_hi_extra,
    )
    target_for_est = Estimand(
        "late",
        u_hi_extra=u_hi_extra,
    )

    if idestimands == "late":
        identified_for_id = [Estimand(esttype="late", u_lo=pscore_lo, u_hi=pscore_hi)]

        identified_for_est = [Estimand(esttype="late")]

    if idestimands == "sharp":
        identified_for_id = [
            Estimand(esttype="cross", dz_cross=(d, z)) for d in [0, 1] for z in [0, 1]
        ]
        identified_for_est = identified_for_id

    u_partition_for_id = np.unique(
        np.array([0, pscore_lo, pscore_hi, pscore_hi + u_hi_extra, 1]),
    )

    if bfunc_type == "bernstein":
        bfuncs = generate_bernstein_basis_funcs(bfunc_options["k_degree"])

    if bfunc_type == "constant":
        bfuncs = generate_constant_splines_basis_funcs(u_partition=u_partition_for_id)

    # ----------------------------------------------------------------------------------
    # Construct DGP for simulation
    # ----------------------------------------------------------------------------------
    # To allow for Bernstein polynomials we need to construct a Bernstein polynomial
    # based DGP that is consistent with the local_ates. In particular, we need to con-
    # struct it such that the true parameter is at the boundary of the identified set,
    # else the simulations will not be informative.
    # To do this, we first construct constant spline MTR functions, which make it easy
    # to set the local ATEs to the desired values. We then use the `identification`
    # function to get the identified set. We then construct the DGP based on the MTRs
    # that generate the upper bound.

    # Fix these for now
    y1_at = 0.75
    y0_at = 0.25

    y1_nt = 0.75
    y0_nt = 0.6

    y1_c = local_ates.complier / 2 + 0.5
    y0_c = -local_ates.complier / 2 + 0.5

    m1_for_id = make_mtr_binary_iv(
        yd_c=y1_c,
        yd_at=y1_at,
        yd_nt=y1_nt,
        pscore_lo=pscore_lo,
        pscore_hi=pscore_hi,
    )
    m0_for_id = make_mtr_binary_iv(
        yd_c=y0_c,
        yd_at=y0_at,
        yd_nt=y0_nt,
        pscore_lo=pscore_lo,
        pscore_hi=pscore_hi,
    )

    res_id = identification(
        target=target_for_id,
        identified_estimands=identified_for_id,
        m0_dgp=m0_for_id,
        m1_dgp=m1_for_id,
        basis_funcs=bfuncs,
        instrument=instrument,
        u_partition=u_partition_for_id,
        **constraints,
    )

    # Generate the DGP based on the MTR solutions for the upper bound in res_id
    m0_for_sim, m1_for_sim = mtr_funcs_from_solution(res=res_id, bound="upper")

    # ----------------------------------------------------------------------------------
    # Perform simulation
    # ----------------------------------------------------------------------------------

    sim_lower_bound = np.zeros(num_sims)
    sim_upper_bound = np.zeros(num_sims)

    sim_ci_lower = np.zeros(num_sims)
    sim_ci_upper = np.zeros(num_sims)

    for i in range(num_sims):
        _data = simulate_data_from_mtrs_binary_iv(
            mtr0=m0_for_sim,
            mtr1=m1_for_sim,
            num_obs=num_obs,
            rng=RNG,
            iv_support=instrument.support,
            iv_pmf=instrument.pmf,
            iv_pscores=instrument.pscores,
        )

        # Perform estimation with confidence intervals
        _res = estimation(
            target=target_for_est,
            identified_estimands=identified_for_est,
            basis_func_type=bfunc_type,
            y_data=_data["y"],
            z_data=_data["z"],
            d_data=_data["d"],
            confidence_interval=confidence_interval,
            confidence_interval_options=confidence_interval_options,
            basis_func_options=bfunc_options,
            **constraints,
        )

        sim_lower_bound[i] = _res.lower_bound
        sim_upper_bound[i] = _res.upper_bound

        sim_ci_lower[i] = _res.ci_lower
        sim_ci_upper[i] = _res.ci_upper

    # Construct DataFrame
    df_res = pd.DataFrame(
        {
            "sim_lower_bound": sim_lower_bound,
            "sim_upper_bound": sim_upper_bound,
            "sim_ci_lower": sim_ci_lower,
            "sim_ci_upper": sim_ci_upper,
        },
    )

    df_res["bfunc_type"] = bfunc_type
    df_res["idestimands"] = idestimands
    df_res["num_sims"] = num_sims
    df_res["num_obs"] = num_obs
    df_res["u_hi_extra"] = u_hi_extra

    for key, val in constraints.items():
        df_res[key] = val

    # TODO(@buddejul): Make this work for subsample size when callable.
    # for key, val in confidence_interval_options.items():

    columns = ["y1_at", "y0_at", "y1_nt", "y0_nt", "y1_c", "y0_c"]
    variables = [y1_at, y0_at, y1_nt, y0_nt, y1_c, y0_c]

    for col, var in zip(columns, variables, strict=True):
        df_res[col] = var

    return df_res
