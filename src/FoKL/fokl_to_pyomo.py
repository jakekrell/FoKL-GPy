import pyomo.environ as pyo
import numpy as np
import warnings
import copy


def _check_models(models):
    """Check 'models' is list of class object(s); return False if cannot be resolved."""
    if not isinstance(models, list):
        if isinstance(models, object):  # single model; make list
            models = [models]
        else:
            return False

    if all(isinstance(model, object) for model in models):
        return models
    else:
        return False


def _check_xvars(xvars):
    """Check 'xvars' is list of list(s) of string(s); return False if cannot be resolved."""
    if isinstance(xvars, str):
        xvars = [[xvars]]
    if isinstance(xvars[0], str):
        xvars = [xvars]

    if isinstance(xvars, list) and isinstance(xvars[0], list) \
            and all(isinstance(xvar[j], str) for xvar in xvars for j in range(len(xvar))):
        return xvars
    else:
        return False


def _check_yvars(yvars):
    """Check 'yvars' is list of string(s); return False if cannot be resolved."""
    if isinstance(yvars, str):
        yvars = [yvars]

    if isinstance(yvars, list) and all(isinstance(yvars[i], str) for i in range(len(yvars))):
        return yvars
    else:
        return False


def fokl_to_pyomo(models, xvars, yvars, draws=None, m=None, xfix=None, yfix=None, truescale=True):
    """

    'to_pyomo' passes inputs to here;
    user may use this for multiple GP's at once (so symbolic bases get defined once, and so 'xvars' can be repeated);

    - 'truescale' changes where 'xfix' gets defined such that 'xfix' for 'truescale=True' must be  entered as true scale;
    - if repeating any 'xvars' across models, the 'truescale' value for the first time it is defined will be used so be
      careful to ensure repeat 'xvars' do not have differently intended 'truescale' values;

    """

    # Check inputs:

    models = _check_models(models)
    if models is False:
        raise ValueError("'models' must be a list of FoKL model class object(s).")

    xvars = _check_xvars(xvars)
    if xvars is False:
        raise ValueError("'xvars' must be a list of list(s) of string(s).")

    yvars = _check_yvars(yvars)
    if yvars is False:
        raise ValueError("'yvars' must be a list of string(s).")

    # Further check inputs:

    n = len(models)

    def _error_align(input_varname):
        raise ValueError(f"'models' and '{input_varname}' must align.")

    if len(xvars) != n or any(len(xvars[i]) != models[i].inputs.shape[1] for i in range(n)):
        _error_align('xvars')

    if len(yvars) != n:
        _error_align('yvars')

    if draws is None:
        draws = []
        for model in models:
            draws.append(model.draws)
    elif isinstance(draws, int):  # then use single 'draws' value for all models
        draws = [draws] * n
    elif isinstance(draws, list):  # then confirm same length as models
        if len(draws) != n:
            _error_align('draws')

    if m is None:
        m = pyo.ConcreteModel()
    else:
        # check for overlapping variable names
        # ...
        # get other pyomo model info
        pass

    if xfix is None:
        xfix = [None] * n
    # else:  # assume properly formatted, e.g., 'xfix=[[0.2, None, 0.6], ..., [None], ..., [0.5, 0.1]]'

    if yfix is None:
        yfix = [None] * n
    # else:  # assume properly formatted, e.g., 'yfix=[342, ..., None, ..., 107]'

    if isinstance(truescale, bool):
        truefalse = copy.copy(truescale)
        truescale = []
        for im in range(n):
            truescale.append([truefalse] * models[im].inputs.shape[1])
    elif isinstance(truescale, list):
        for im in range(n):
            if isinstance(truescale[im], bool):
                truescale[im] = [truescale[im]] * models[im].inputs.shape[1]
    if n == 1:
        truescale = [truescale]
    if any(truescale[im] for im in range(n)) is True:
        xvars_true = copy.deepcopy(xvars)
        i_norm = []  # indices of true scale input variables in all models
        for im in range(n):
            i_norm_im = []  # indices of true scale input variables in current model
            for j in range(models[im].inputs.shape[1]):
                if truescale[im][j] is True:
                    i_norm_im.append(j)
                    xvars[im][j] = f"{xvars[im][j]}_normalized"  # only applied to not truescale models
            i_norm.append(i_norm_im)

    for model in models:
        try:
            if model.kernel != 'Bernoulli Polynomials':
                warnings.warn("'kernel' should be 'Bernoulli Polynomials'. The kernel is being switched for Pyomo but "
                              "it is highly recommended to retrain the model.", category=UserWarning)
        except Exception as exception:
            pass  # assume user did not train model but is manually passing 'betas', 'mtx', 'draws' in model(s)

    # Loop through models:

    im = -1  # index of model, for indexing 'xvars' and 'yvars'
    for model in models:
        im += 1

        # Convert FoKL to Pyomo:

        t = np.array(model.mtx - 1, dtype=int)  # indices of polynomial (where 0 is B1 and -1 means none)
        lt = t.shape[0] + 1  # length of terms (including beta0)
        lv = t.shape[1]  # length of input variables

        ni_ids = []  # orders of basis functions used (where 0 is B1), per term
        basis_n = []  # for future use when indexing 'm.fokl_basis'
        for j in range(lv):  # for input variable in input variables
            ni_ids.append(np.sort(np.unique(t[:, j][t[:, j] != -1])).tolist())
            basis_n += ni_ids[j]
        # n_ids = np.sort(np.unique(basis_n))  # orders of basis functions used (where 0 is B1), total (not used)

        m.add_component(f"GP{im}_scenarios", pyo.Set(initialize=range(draws[im])))  # index for scenario (i.e., FoKL draw)
        m.add_component(f"GP{im}_j", pyo.Set(initialize=range(lv)))  # index for FoKL input variable

        # Define FoKL inputs and output:

        if m.find_component(yvars[im]) is None:  # then define; else a previous model already defined this variable
            m.add_component(yvars[im], pyo.Var(m.component(f"GP{im}_scenarios"), within=pyo.Reals))  # FoKL output
        basis_nj = []
        for j in m.component(f"GP{im}_j"):
            if m.find_component(xvars[im][j]) is None:  # then define; else a previous model already defined this variable
                m.add_component(xvars[im][j], pyo.Var(within=pyo.Reals, bounds=[0, 1], initialize=0.0))  # FoKL input variables
                if truescale[im][j] is True:  # create expression relating normalized variable to true scale
                    m.add_component(xvars_true[im][j], pyo.Var(within=pyo.Reals, bounds=model.normalize[j], initialize=model.normalize[j][0]))

                    # Add normalization constraint:

                    if m.find_component(f"GP{im}_normalize") is None:  # reduce redundancy if repeat 'xvars' (may be extra constraint indices if some variables repeat but not others)
                        m.add_component(f"GP{im}_normalize", pyo.Constraint(i_norm[im]))

                    def symbolic_normalize(m):
                        """Relate normalized and true scale input variable."""
                        m.component(f"GP{im}_normalize")[j] = m.component(xvars_true[im][j]) == m.component(xvars[im][j]) * (model.normalize[j][1] - model.normalize[j][0]) + model.normalize[j][0]
                        return

                    symbolic_normalize(m)  # may be better to write as rule

            for n in ni_ids[j]:  # for order of basis function in unique orders, per current input variable 'm.~x[j]'
                basis_nj.append([n, j])

        # Define FoKL model:

        def symbolic_basis(m):
            """Basis functions as symbolic. See 'evaluate_basis' for source of equation."""
            for [n, j] in basis_nj:
                m.component(f"GP{im}_basis")[n, j] = model.phis[n][0] + sum(model.phis[n][k] * (m.component(xvars[im][j]) ** k)
                                                           for k in range(1, len(model.phis[n])))
            return

        m.add_component(f"GP{im}_basis", pyo.Expression(basis_nj))  # create indices ONLY for used basis functions
        symbolic_basis(m)  # may be better to write as rule, but 'pyo.Expression(basis_nj, rule=symbolic_basis)' failed

        m.add_component(f"GP{im}_k", pyo.Set(initialize=range(lt)))  # index for FoKL term (where 0 is beta0)
        m.add_component(f"GP{im}_b", pyo.Var(m.component(f"GP{im}_scenarios"), m.component(f"GP{im}_k")))  # FoKL coefficients (i.e., betas)
        for i in m.component(f"GP{im}_scenarios"):  # for scenario (i.e., draw) in scenarios (i.e., draws)
            for k in m.component(f"GP{im}_k"):  # for term in terms
                m.component(f"GP{im}_b")[i, k].fix(model.betas[-(i + 1), k])  # define values of betas, with y[0] as last FoKL draws

        def symbolic_fokl(m):
            """FoKL models (i.e., scenarios) as symbolic, assuming 'Bernoulli Polynomials."""
            for i in m.component(f"GP{im}_scenarios"):  # for scenario (i.e., draw) in scenarios (i.e., draws)
                m.component(f"GP{im}_expr")[i] = m.component(f"GP{im}_b")[i, 0]  # initialize with beta0
                for k in range(1, lt):  # for term in non-zeros terms (i.e., exclude beta0)
                    tk = t[k - 1, :]  # interaction matrix of current term
                    tk_mask = tk != -1  # ignore if -1 (recall -1 basis function means none)
                    if any(tk_mask):  # should always be true because FoKL 'fit' removes rows from 'mtx' without basis
                        term_k = m.component(f"GP{im}_b")[i, k]
                        for j in m.component(f"GP{im}_j"):  # for input variable in input variables
                            if tk_mask[j]:  # for variable in term
                                term_k *= m.component(f"GP{im}_basis")[tk[j], j]  # multiply basis function(s) with beta to form term
                    else:
                        term_k = 0
                    m.component(f"GP{im}_expr")[i] += term_k  # add term to expression
            return

        m.add_component(f"GP{im}_expr", pyo.Expression(m.component(f"GP{im}_scenarios")))  # FoKL models (i.e., scenarios, draws)
        symbolic_fokl(m)  # may be better to write as rule

        def symbolic_scenario(m):
            """Define each scenario, meaning a different draw of 'betas' for y=f(x), as a constraint."""
            for i in m.component(f"GP{im}_scenarios"):
                m.component(f"GP{im}_constr")[i] = m.component(yvars[im])[i] == m.component(f"GP{im}_expr")[i]
            return

        m.add_component(f"GP{im}_constr", pyo.Constraint(m.component(f"GP{im}_scenarios")))  # set of constraints, one per scenario
        symbolic_scenario(m)  # may be better to write as rule

        if xfix[im] is not None:
            for j in m.component(f"GP{im}_j"):
                if xfix[im][j] is not None:
                    if truescale[im][j] is True:
                        m.component(xvars_true[im][j]).fix(xfix[im][j])
                    else:
                        m.component(xvars[im][j]).fix(xfix[im][j])
        if yfix[im] is not None:
            for i in m.component(f"GP{im}_scenarios"):
                m.component(yvars[im])[i].fix(yfix[im])

    # Return Pyomo model with all FoKL models embedded:

    return m

