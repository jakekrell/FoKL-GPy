"""
Functions for interfacing a GP model from a FoKL class with a Pyomo model.

| Function      | Description                                                                      |
|---------------|----------------------------------------------------------------------------------|
| fix_betas     | set values of the GP's beta coefficients defined as variables in the Pyomo model |
| fokl_to_pyomo | embed GP from FoKL model into Pyomo model                                        |

"""
import numpy as np
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.environ import *
import warnings


# Internal functions:

def _process_arguments(self, xvars, yvar, m, draws, t_span, mtx, betas, minmax):
    """Process/format input arguments for 'to_pyomo' method."""
    if m is None:
        m = pyo.ConcreteModel('Global')

    if m.find_component('t') is None:
        if t_span is None:  # ODE not requested, so single index to avoid if-else statements in internal code
            m.t = pyo.Set(initialize=range(1))
        elif isinstance(t_span, list):
            m.t = dae.ContinuousSet(bounds=t_span)
        else:
            raise ValueError("Argument 't_span' must be a list of integration bounds.")
    elif t_span is not None:
        warnings.warn("Ignoring argument 't_span' because 'm.t' is already defined.", category=UserWarning)

    if not isinstance(xvars, list):  # if not list, make list
        xvars = [xvars]
    if isinstance(yvar, list):  # if list, make not list
        yvar = yvar[0]

    if minmax is None:
        minmax = self.minmax
        
    j = -1
    for xvar in xvars:
        j += 1
        if isinstance(xvar, str):  # else pre-defined Pyomo component, so ignore
            m.add_component(xvar, pyo.Var(m.t, domain=pyo.Reals, bounds=minmax[j]))
            xvars[j] = m.component(xvar)

    if isinstance(yvar, str):  # else pre-defined Pyomo component, so ignore
        m.add_component(yvar, pyo.Var(m.t, domain=pyo.Reals))
        yvar = m.component(yvar)

    if draws is None:
        draws = self.draws

    if mtx is None:
        mtx = self.mtx

    if betas is None:
        betas = self.betas

    return self, xvars, yvar, m, draws, t_span, mtx, betas, minmax


def _gp_as_pyomo(name, tvec, phis, draws, mtx, betas, xvars, minmax, model=None, scenarios=None):
    """tvec == m.t"""
    # Initialize sub-model for GP:
    if model is None:
        mGP = pyo.ConcreteModel(name)
        _with_block = True
    else:
        mGP = model
        _with_block = False

    # Some constants:
    mtx = np.array(mtx, dtype=int)  # indices/orders of basis functions (where 1 is B1 and 0 means none)

    # Some sets:
    mGP.terms = pyo.Set(initialize = range(mtx.shape[0] + 1))  # terms (including beta0)
    mGP.orders = pyo.Set(initialize = np.unique(mtx[mtx != 0]))  # orders of basis functions
    mGP.attributes = pyo.Set(initialize = range(mtx.shape[1]))  # input variables
    if scenarios is not None:
        if len(scenarios) == draws:
            raise NotImplementedError("Currently, length of 'scenarios' must equal 'draws'. In other words, 'scenarios' must index each of FoKL's 'draws'.")
        mGP.draws = scenarios  # using 'scenarios' instead of 'range(draws)' allows 'm.s' to be strings, etc.
    else:  # if 'scenarios' is None
        mGP.draws = pyo.Set(initialize = range(draws))

    # if len(scenarios) == 1 or scenarios is None:  # then average draws into single Pyomo scenario
    # elif len(scenarios) == draws:  # then each draw is scenario
    # else:
    #     raise NotImplementedError()

    # COMMENTS:
    #   - if s None
    #       - m.s_temp = range(1)  # placeholder index
    #   - if len(s or m.s_temp) == 1  # (and draws != 1 ... else require draws > 1)
    #       - avg draws into single scenario
    #   - elif len(s) == draws
    #       - then each draw is scenario
    #   - else not implemented

    # ================
    # RTW:


    # Define beta coefficients:
    mGP.beta = pyo.Param(mGP.draws, mGP.terms, mutable=True)  # mutable=True, to change the value dynamically
    mGP.beta_avg = pyo.Param(mGP.terms, mutable=True)
    fix_betas(mGP, betas)

    # Define expression of normalized attributes (i.e., input variables):
    
    # TWO VERSIONS OF EQ_NORM / OR SWITCH CASE IN SINGLE FUNC:

    def _eq_norm(mGP, t, j):
        """Normalization constraint. (scenarios is None) ... modify slightly to index m.s_temp==1"""
        return (xvars[j][t] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])
    
    def _eq2_norm(mGP, t, j):
        """Normalization constraint. (scenarios is not None) ... index xvars by m.s"""
        return (xvars[j][t] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    # mGP.x = pyo.Expression(tvec, mGP.attributes, rule=_eq_norm)
    mGP.x = pyo.Expression(tvec, scenarios, mGP.attributes, rule=_eq_norm)

    # ===================================================================
    # Define polynomials (i.e., "basis" functions):
    
    nj = []  # list of [order, attribute] combinations used in GP
    for attribute in mGP.attributes:
        orders_j = np.unique(mtx[:, attribute])
        if any(orders_j != 0):
            for order_j in orders_j[orders_j != 0]:
                nj.append([order_j, attribute])
    
    # X NEEDS TO BE INDEXED BY m.s

    def _eq_phi(mGP, t, n, j):
        """FoKL's 'basis' functions."""
        nm1 = n - 1  # Python indexing, since n=1 refers to B1 which is phis[0]
        return phis[nm1][0] + sum(phis[nm1][k] * mGP.x[t, j] ** k for k in range(1, len(phis[nm1])))

    # INDEX BY m.s

    mGP.phi = pyo.Expression(tvec, nj, rule=_eq_phi)

    # ===================================================================
    # Build GP expression:

    # Draws:

    # CONSIDER AVERAGING ALL TO PLACE IN SINGLE SCENARIO, OR EACH DRAW IN EACH SCENARIO (IN WHICH CASE NO Y_AVG NEEDED)

    def _eq_y(mGP, t, draw):
        """FoKL's GP equation."""
        y = mGP.beta[draw, 0]  # initialize
        
        for term in range(1, len(mGP.terms)):  # == m.terms[1::]
            y_term = mGP.beta[draw, term]

            for j in mGP.attributes:
                n = mtx[term - 1, j]

                if n != 0:  # since 0 means none
                    y_term *= mGP.phi[t, n, j]

            y += y_term

        return y

    mGP.y = pyo.Expression(tvec, mGP.draws, rule=_eq_y)

    # Average (IGNORE FOR NOW BECAUSE m.s):

    # def _eq_y_avg(mGP, t):
    #     """FoKL's GP equation, averaged across draws."""
    #     y = mGP.beta_avg[0]  # initialize
        
    #     for term in range(1, len(mGP.terms)):  # == m.terms[1::]
    #         y_term = mGP.beta_avg[term]

    #         for j in mGP.attributes:
    #             n = mtx[term - 1, j]

    #             if n != 0:  # since 0 means none
    #                 y_term *= mGP.phi[t, n, j]

    #         y += y_term

    #     return y

    # mGP.y_avg = pyo.Expression(tvec, rule=_eq_y_avg)

    # Standard deviation (IGNORE FOR NOW):

    # def _eq_y_std(mGP, t):
    #     """Standard deviation of draws from FoKL's GP equation."""
    #     return sqrt(sum(mGP.y[t, draw] ** 2 for draw in mGP.draws) / len(mGP.draws) + 1e-9)

    # mGP.y_std = pyo.Expression(tvec, rule=_eq_y_std)

    return mGP


# End internal functions.
# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
# =============================================================================================================
# Module functions:

def fix_betas(m, betas, GPi_draws=None, i=0):
    """
    Fix the already-initialized Pyomo beta Param's to scalar values in 'betas', using last 'betas' draw as first Pyomo draw. Include average.
    
    | Argument  | Type          | Description                                     |
    |-----------|---------------|-------------------------------------------------|
    | m         | ConcreteModel | Pyomo model containing embedded GP              |
    | betas     | ndarray       | [draws x terms] beta coefficients of FoKL model |
    | GPi_draws | Set           | index of 'm.GP{i}_beta[draws, terms]'           |
    | i         | int           | index of GP to update                           |
    
    | Output | Type          | Description                                  |
    |--------|---------------|----------------------------------------------|
    | m      | ConcreteModel | Pyomo model with 'm.GP{i}_beta' values fixed |

    """
    if GPi_draws is None:
        GPi_draws = m.component(f"GP{i}_draws")

    i_draw = -1  # index of draw (to index betas ndarray)
    for draw in GPi_draws:
        i_draw += 1
        for term in m.component(f"GP{i}_terms"):
            m.component(f"GP{i}_beta")[draw, term] = betas[-(i_draw + 1), term]

    return m


def fokl_to_pyomo(self, xvars, yvar, m=None, t=None, draws=None, mtx=None, betas=None, minmax=None):
    """
    Convert GP model from FoKL class to Pyomo model.
    
    | Arg.   | Type                             | Description                                                                                                                                          | Default           |
    |--------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
    | self   | FoKL class                       | FoKL class object with Bernoulli Polynomials kernel, 'FoKLRoutines.FoKL(kernel=1)'                                                                   | -                 |
    | xvars  | list of Pyomo component(s)       | list of GP input variable(s), i.e., attributes; can be pyo.Var, dae.DerivativeVar, etc.; when defining, index by 't' and/or 'draws' where applicable | -                 |
    | yvar   | Pyomo component                  | GP output variable; behaves like 'xvars'                                                                                                             | -                 |
    | m      | Pyomo model                      | pre-defined Pyomo model                                                                                                                              | pyo.ConcreteModel |
    | t      | dae.ContinuousSet                | index for integration time to achieve ODE functionality, i.e., 't = dae.ContinuousSet(bounds=[t0, tf])'                                              | None              |
    | draws  | int or pyo.Set                   | number of GP draws to embed in Pyomo, or Set indexing draws                                                                                          | self.draws        |
    | mtx    | ndarray [terms - 1 x attributes] | GP's interaction matrix                                                                                                                              | self.mtx          |
    | betas  | ndarray [draws x terms]          | GP's coefficients                                                                                                                                    | self.betas        |
    | minmax | list of lists of two floats      | GP's normalization of input variables [[min, max], ..., [min, max]]                                                                                  | self.minmax       |
    
    | Output | Type        | Description                                        |
    |--------|-------------|----------------------------------------------------|
    | m      | Pyomo model | input argument 'm' with 'self' embedded as 'm.GP#' |
    
    Reserved Pyomo components:
        - 'm.GP#', where # is integer index of GP beginning at 0; reservation includes all 'm.GP#_{str}' components too
        
    Tips:
        - 'xvars' gets bounded by 'minmax' of FoKL model 'self';
          this is not technically required since the GP is a polynomial, but is recommended because the GP is not intended to extrapolate.
            - 'm.[xvars[j]].setlb(GP.minmax[j][0])'
            - 'm.[xvars[j]].setub(GP.minmax[j][1])'
        - 'yvar' indices are [t, draws] if passing in Pyomo components for 't', 'draws'; 'xvars' indices are assumed to be either [t], [draws], [t, draws], or none
    
    """
    # Process defaults:
    if m is None:
        m = pyo.ConcreteModel("FoKL-to-Pyomo Model")
    if draws is None:
        draws = self.draws
    if mtx is None:
        mtx = self.mtx
    if betas is None:
        betas = self.betas
    if minmax is None:
        minmax = self.minmax
    
    # Check for type errors:
    if not isinstance(xvars, list):
        raise TypeError()
    for xvar in xvars:
        if not any(isinstance(xvar, type) for type in [pyo.Var, dae.DerivativeVar]):
            raise TypeError()
    if not any(isinstance(yvar, type) for type in [pyo.Var, dae.DerivativeVar]):
        raise TypeError()
    if not isinstance(m, pyo.ConcreteModel):
        raise TypeError()
    if not (t is None or isinstance(t, dae.ContinuousSet)):
        raise TypeError()
    if not any(isinstance(draws, type) for type in [int, pyo.Set]):
        raise TypeError()
    if not isinstance(mtx, np.ndarray):
        raise TypeError()
    if not isinstance(betas, np.ndarray):
        raise TypeError()
    if not isinstance(minmax, list):
        raise TypeError()
    for minmax_i in minmax:
        if not isinstance(minmax_i, list):
            raise TypeError()
        if not any(isinstance(minmax_i[i], float) for i in range(2)):
            raise TypeError()

    # Warn about index assumptions regarding time ('t') and/or scenarios ('draws'):
    for var in [yvar] + xvars:
        if var.dim() > 2:
            raise NotImplementedError("Pyomo variables indexed by more than time ('t') and scenarios ('draws') are not supported.")
    td = [isinstance(t, dae.ContinuousSet), isinstance(draws, pyo.Set)]
    i_td = [[False, False]] * (1 + len(xvars))  # boolean array for how each variable in 'xvars + [yvar]' gets indexed; corresponds to ['t', 'draws'] indices
    if all(td):
        warnings.warn(f"Assuming '{yvar.name}' indexed by ['{t.name}', '{draws.name}'].", category=SyntaxWarning)
        i_td[-1] = [True, True]  # yvar indexed by ['t', 'draws']
        _warn_equal_len = True
        i = -1
        for xvar in xvars:
            i += 1
            if xvar.dim() == 2:
                warnings.warn(f"Assuming '{xvar.name}' indexed by ['{t.name}', '{draws.name}'].", category=SyntaxWarning)
                i_td[i] = [True, True]  # xvar indexed by ['t', 'draws']
            elif xvar.dim() == 1:
                if len(t) == len(draws):
                    if _warn_equal_len:
                        warnings.warn(f"Pyomo sets '{t.name}' and '{draws.name}' are indistinguishable due to equal length; assuming '{t.name}' as the index for variables with one dimension.", category=SyntaxError)
                    _warn_equal_len = False
                if len(xvar) == len(t):
                    warnings.warn(f"Assuming '{xvar.name}' indexed by '{t.name}'.", category=SyntaxWarning)
                    i_td[i][0] = True  # xvar indexed by 't'
                elif len(xvar) == len(draws):
                    warnings.warn(f"Assuming '{xvar.name}' indexed by '{draws.name}'.", category=SyntaxWarning)
                    i_td[i][1] = True  # xvar indexed by 'draws'
                else:
                    raise NotImplementedError(f"'{xvar.name}' may only be indexed by '{t.name}' and/or '{draws.name}'.")
            else:
                warnings.warn(f"Assuming '{xvar.name}' not indexed.", category=SyntaxWarning)
    elif td[0]:
        warnings.warn(f"Assuming '{yvar.name}' indexed by '{t.name}'.", category=SyntaxWarning)
        i_td[-1][0] = True  # yvar indexed by 't'
        i = -1
        for xvar in xvars:
            i += 1
            if xvar.dim() == 1:
                warnings.warn(f"Assuming '{xvar.name}' indexed by '{t.name}'.", category=SyntaxWarning)
                i_td[i][0] = True  # xvar indexed by 't'
            else:
                warnings.warn(f"Assuming '{xvar.name}' not indexed.", category=SyntaxWarning)
    elif td[1]:
        warnings.warn(f"Assuming '{yvar.name}' indexed by '{draws.name}'.", category=SyntaxWarning)
        i_td[-1][1] = True  # yvar indexed by 'draws'
        i = -1
        for xvar in xvars:
            i += 1
            if xvar.dim() == 1:
                warnings.warn(f"Assuming '{xvar.name}' indexed by '{draws.name}'.", category=SyntaxWarning)
                i_td[i][1] = True  # xvar indexed by 'draws'
            else:
                warnings.warn(f"Assuming '{xvar.name}' not indexed.", category=SyntaxWarning)

    # Find next available GP index:
    i = 0
    while m.find_component(f"GP{i}") is not None:
        i += 1

    # Some constants:
    mtx = np.array(mtx, dtype=int)  # indices/orders of basis functions (where 1 is B1 and 0 means none)

    # Some sets:
    m.add_component(f"GP{i}_terms", pyo.Set(initialize = range(mtx.shape[0] + 1)))    # terms (including beta0)
    m.add_component(f"GP{i}_orders", pyo.Set(initialize = np.unique(mtx[mtx != 0])))  # orders of basis functions
    m.add_component(f"GP{i}_attributes", pyo.Set(initialize = range(mtx.shape[1])))   # indices of input variables
    if isinstance(draws, int):
        m.add_component(f"GP{i}_draws", pyo.Set(initialize = range(draws)))           # draws (via int)
        GPi_draws = m.component(f"GP{i}_draws")
    else:
        GPi_draws = draws                                                             # draws (via pre-defined scenarios, i.e., 'draws' Set); could be str's, etc.
    GPi_terms = m.component(f"GP{i}_terms")
    GPi_orders = m.component(f"GP{i}_orders")
    GPi_attributes = m.component(f"GP{i}_attributes")

    # Define beta coefficients:
    m.add_component(f"GP{i}_beta", pyo.Param(GPi_draws, GPi_terms, mutable=True))  # 'mutable=True', to change the values dynamically
    GPi_beta = m.component(f"GP{i}_beta")

    # Average beta coefficients:
    
    def _beta_avg(m, term):
        """Average beta Param's rather than 'yvar' Expression's to yield faster Pyomo solutions.
        Defining a 'beta_avg' Param would be faster and would not require 'yvar' draws, though this optional feature is left for future development."""
        return sum(GPi_beta[draw, term] for draw in GPi_draws) / len(GPi_draws)
    
    m.add_component(f"GP{i}_beta_avg", pyo.Expression(GPi_terms, rule=_beta_avg))    
    fix_betas(m, betas, GPi_draws)

    # Define switch cases of normalized attributes:

    def _eq_norm_00(m):
        """Normalization constraint; no 't', no 'draws')."""
        return (xvars[j] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    def _eq_norm_01(m, draw):
        """Normalization constraint; no 't', yes 'draws'."""
        return (xvars[j][draw] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    def _eq_norm_10(m, t_ind):
        """Normalization constraint; yes 't', no 'draws'."""
        return (xvars[j][t_ind] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    def _eq_norm_11(m, t_ind, draw):
        """Normalization constraint; yes 't', yes 'draws'."""
        return (xvars[j][t_ind, draw] - minmax[j][0]) / (minmax[j][1] - minmax[j][0])

    for j in GPi_attributes:
        if i_td[j] == [False, False]:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(rule=_eq_norm_00))
        elif i_td[j] == [False, True]:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(GPi_draws, rule=_eq_norm_01))
        elif i_td[j] == [True, False]:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(t, rule=_eq_norm_10))
        elif i_td[j] == [True, True]:
            m.add_component(f"GP{i}_x{j}", pyo.Expression(t, GPi_draws, rule=_eq_norm_11))

    

    # ===============================================================================================================================
    # ===============================================================================================================================
    # ===============================================================================================================================
    # ===============================================================================================================================
    # ===============================================================================================================================
    # ===============================================================================================================================
    # ===============================================================================================================================
    # RETURN TO WORK (RTW):

    return



    # ===================================================================
    # Define polynomials (i.e., "basis" functions):
    
    nj = []  # list of [order, attribute] combinations used in GP
    for attribute in mGP.attributes:
        orders_j = np.unique(mtx[:, attribute])
        if any(orders_j != 0):
            for order_j in orders_j[orders_j != 0]:
                nj.append([order_j, attribute])
    
    # X NEEDS TO BE INDEXED BY m.s

    def _eq_phi(mGP, t, n, j):
        """FoKL's 'basis' functions."""
        nm1 = n - 1  # Python indexing, since n=1 refers to B1 which is phis[0]
        return phis[nm1][0] + sum(phis[nm1][k] * mGP.x[t, j] ** k for k in range(1, len(phis[nm1])))

    # INDEX BY m.s

    mGP.phi = pyo.Expression(tvec, nj, rule=_eq_phi)







    # if isinstance(t, dae.ContinuousSet):
    #     for xvar in xvars:
            






    # if with_blocks is True and m is None:
    #     raise NotImplementedError()

    # # Process input arguments:
    # self, xvars, yvar, m, draws, t_span, mtx, betas, minmax = _process_arguments(self, xvars, yvar, m, draws, t_span, mtx, betas, minmax)

    # # Find next available GP index:
    # if with_blocks is True:
    #     i = 0
    #     while m.find_component(f"GP{i}") is not None:
    #         i += 1

    # # Create Pyomo model with GP:
    # if with_blocks is False:
    #     model = m
    #     gp_name = 'GP Model'
    # else:
    #     model = None
    #     gp_name = f"GP{i}"
    # mGP = _gp_as_pyomo(gp_name, m.t, self.phis, draws, mtx, betas, xvars, minmax, model=model, scenarios=scenarios)

    # # Set 'yvar' equal to GP:
    
    # # IF s IS NONE --> yvar not indexed by m.s; else is; etc.  ---> maybe two versions of constr.

    # def _constr_yvar(mGP, t):
    #     """Set 'yvar' equal to GP."""
    #     return yvar[t] == mGP.y_avg[t]
    
    # mGP.constr_yvar = pyo.Constraint(m.t, rule=_constr_yvar)

    # # Merge 'mGP' with global Pyomo model:
    # if with_blocks is True:
    #     m.add_component(f"GP{i}", mGP)

    #     return m

    # else:

    #     return mGP
    


    # # -----------------
    # #TODO
    # #   - reproduce with s is None
    # #   - then try s (with all xvars as [m.t, m.s])




