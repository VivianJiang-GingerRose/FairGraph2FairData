import itertools
import math
import os
from warnings import warn

import networkx as nx
import numpy as np
from joblib import Parallel, delayed

from FairnessAwareHC import HAS_PANDAS
from FairnessAwareHC.inference import Inference
from FairnessAwareHC.utils import _check_1d_array_object, _check_length_equal

if HAS_PANDAS:
    import pandas


class BayesianModelInference(Inference):
    """
    Inference class specific to Bayesian Models
    """

    def __init__(self, model):
        """
        Class to calculate probability (pmf) values specific to Bayesian Models

        Parameters
        ----------
        model: Bayesian Model
            model on which inference queries will be computed
        """
        from FairnessAwareHC.models import BayesianNetwork

        if not isinstance(model, BayesianNetwork):
            raise TypeError(
                f"Model expected type: BayesianNetwork, got type: {type(model)}"
            )
        super(BayesianModelInference, self).__init__(model)
        self._initialize_structures()

        self.topological_order = list(nx.topological_sort(model))

    def pre_compute_reduce(self, variable):
        """
        Get probability arrays for a node as function of conditional dependencies

        Internal function used for Bayesian networks, eg. in BayesianModelSampling
        and BayesianModelProbability.

        Parameters
        ----------
        variable: Bayesian Model Node
            node of the Bayesian network

        Returns
        -------
        dict: dictionary with probability array for node
            as function of conditional dependency values
        """
        variable_cpd = self.model.get_cpds(variable)
        variable_evid = variable_cpd.variables[:0:-1]
        cached_values = {}

        for state_combination in itertools.product(
            *[range(self.cardinality[var]) for var in variable_evid]
        ):
            states = list(zip(variable_evid, state_combination))
            cached_values[state_combination] = variable_cpd.reduce(
                states, inplace=False, show_warnings=False
            ).values

        return cached_values

    @staticmethod
    def _reduce(variable_cpd, variable_evid, sc_values):
        """
        Method to compute values of the `variable_cpd` when it it reduced on
        `variable_evid` with states `sc_values`. This is a stripped down
        version DiscreteFactor.reduce to only compute the values for faster
        runtime.

        Parameters
        ----------
        variable_cpd: Instance of FairnessAwareHC.factors.discrete.TabularCPD
            The CPD that will be reduced.

        variable_evid: list
            List of variable name that need to be reduced.

        sc_values: list
            list of list of states (corresponding to variable_evid) to which to
            reduce the CPD.

        Returns
        -------
        list: List of np.array with each element representing the reduced
                values correponding to the states in sc_values.
        """
        return_values = []
        for sc in sc_values:
            sc = list(zip(variable_evid, sc))
            try:
                values = [
                    (var, variable_cpd.get_state_no(var, state_name))
                    for var, state_name in sc
                ]
            except KeyError:
                values = sc
            slice_ = [slice(None)] * len(variable_cpd.variables)
            for var, state in values:
                var_index = variable_cpd.variables.index(var)
                slice_[var_index] = state
            return_values.append(variable_cpd.values[tuple(slice_)])
        return return_values

    def pre_compute_reduce_maps(self, variable, state_combinations=None, n_jobs=-1):
        """
        Get probability array-maps for a node as function of conditional dependencies

        Internal function used for Bayesian networks, eg. in BayesianModelSampling
        and BayesianModelProbability.

        Parameters
        ----------
        variable: Bayesian Model Node
            node of the Bayesian network

        state_combinations: list (default=None)
            List of tuple of state combinations for which to compute the reductions maps.

        n_jobs: int (default: -1)
            The number of CPU cores to use. By default uses all.

        Returns
        -------
        dict: dictionary with probability array-index for node as function of conditional dependency values,
            dictionary with mapping of probability array-index to probability array.
        """
        variable_cpd = self.model.get_cpds(variable)
        variable_evid = variable_cpd.variables[:0:-1]

        if state_combinations is None:
            state_combinations = [
                tuple(sc)
                for sc in itertools.product(
                    *[range(self.cardinality[var]) for var in variable_evid]
                )
            ]

        # Comptue batch sizes and call _reduce in parallel.
        if n_jobs == -1:
            n_jobs = os.cpu_count()
        batch_size = math.ceil(len(state_combinations) / n_jobs)

        weights_list = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(BayesianModelInference._reduce)(
                variable_cpd,
                variable_evid,
                state_combinations[
                    (batch_size * i) : min(
                        batch_size * (i + 1), len(state_combinations)
                    )
                ],
            )
            for i in range((len(state_combinations) // batch_size) + 1)
        )

        weights_list = np.array(list(itertools.chain(*weights_list)))
        unique_weights, weights_indices = np.unique(
            weights_list, axis=0, return_inverse=True
        )

        # convert weights to index; make mapping of state to index
        state_to_index = dict(zip(state_combinations, weights_indices))

        # make mapping of index to weights
        index_to_weight = dict(enumerate(unique_weights))

        # return mappings of state to index, and index to weight
        return state_to_index, index_to_weight


class BaseGradLogPDF(object):
    """
    Base class for evaluating gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize  self.grad_log and self.log_pdf

    Parameters
    ----------
    variable_assignments : A 1d array like object (numpy.ndarray or list)
        Vector representing values(assignments) of variables at which we want to find gradient and log

    model : An instance of FairnessAwareHC.models

    Examples
    --------
    >>> from FairnessAwareHC.factors import GaussianDistribution
    >>> from FairnessAwareHC.inference.continuous import BaseGradLogPDF
    >>> import numpy as np
    >>> class GradLogGaussian(BaseGradLogPDF):
    ...     def __init__(self, position, model):
    ...         BaseGradLogPDF.__init__(self, position, model)
    ...         self.grad_log, self.log_pdf = self._get_gradient_log_pdf()
    ...     def _get_gradient_log_pdf(self):
    ...         sub_vec = self.position - self.model.mean.flatten()
    ...         grad = - np.dot(self.model.precision_matrix, sub_vec)
    ...         log_pdf = 0.5 * float(np.dot(sub_vec, grad))
    ...         return grad, log_pdf
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.2], [0.2, 7]])
    >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
    >>> dist_param = np.array([0.1, 0.9])
    >>> grad_logp, logp = GradLogGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([ 0.90229885, -0.01149425])
    """

    def __init__(self, variable_assignments, model):
        self.variable_assignments = _check_1d_array_object(
            variable_assignments, "variable_assignments"
        )
        _check_length_equal(
            variable_assignments,
            model.variables,
            "variable_assignments",
            "model.variables",
        )

        self.model = model

        # The gradient log of probability distribution at position
        self.grad_log = None

        # The gradient log of probability distribution at position
        self.log_pdf = None

    def get_gradient_log_pdf(self):
        """
        Returns the gradient log and log of model at given position

        Returns
        -------
        Returns a tuple of following types

        numpy.array: Representing gradient log of model at given position

        float: Representing log of model at given position

        Example
        --------
        >>> # Using implementation of GradLogPDFGaussian
        >>> from FairnessAwareHC.sampling.base import GradLogPDFGaussian
        >>> from FairnessAwareHC.factors.continuous import GaussianDistribution
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, -5], [-5, 2]])
        >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
        >>> dist_param = np.array([0.6, 0.8])
        >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
        >>> logp
        0.025217391304347823
        >>> grad_logp
        array([-0.07826087, -0.09565217])
        """
        return self.grad_log, self.log_pdf


class GradLogPDFGaussian(BaseGradLogPDF):
    """
    Class for finding gradient and gradient log of Joint Gaussian Distribution
    Inherits FairnessAwareHC.inference.base_continuous.BaseGradLogPDF
    Here model must be an instance of GaussianDistribution

    Parameters
    ----------
    variable_assignments : A 1d array like object (numpy.ndarray or list)
        Vector representing values of variables at which we want to find gradient and log

    model : An instance of FairnessAwareHC.models.GaussianDistribution

    Example
    -------
    >>> from FairnessAwareHC.sampling import GradLogPDFGaussian
    >>> from FairnessAwareHC.factors.continuous import GaussianDistribution
    >>> import numpy as np
    >>> mean = np.array([3, 4])
    >>> covariance = np.array([[5, 4], [4, 5]])
    >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
    >>> dist_param = np.array([12, 21])
    >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -34.777777777777771
    >>> grad_logp
    array([ 2.55555556, -5.44444444])
    """

    def __init__(self, variable_assignments, model):
        BaseGradLogPDF.__init__(self, variable_assignments, model)
        self.grad_log, self.log_pdf = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at position
        """
        sub_vec = self.variable_assignments - self.model.mean.flatten()
        grad = -np.dot(self.model.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.dot(sub_vec, grad)

        return grad, log_pdf


class BaseSimulateHamiltonianDynamics(object):
    """
    Base class for proposing new values of position and momentum by simulating Hamiltonian Dynamics.

    Classes inheriting this base class can be passed as an argument for
    simulate_dynamics in inference algorithms.

    Parameters
    ----------
    model : An instance of FairnessAwareHC.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf : A subclass of FairnessAwareHC.inference.continuous.BaseGradLogPDF
        A class for finding gradient log and log of distribution

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Examples
    --------
    >>> from FairnessAwareHC.sampling import BaseSimulateHamiltonianDynamics
    >>> from FairnessAwareHC.factors.continuous import GaussianDistribution
    >>> from FairnessAwareHC.sampling import GradLogPDFGaussian
    >>> import numpy as np
    >>> # Class should initialize self.new_position, self.new_momentum and self.new_grad_logp
    >>> # self.new_grad_logp represents gradient log at new proposed value of position
    >>> class ModifiedEuler(BaseSimulateHamiltonianDynamics):
    ...     def __init__(self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None):
    ...         BaseSimulateHamiltonianDynamics.__init__(self, model, position, momentum,
    ...                                                  stepsize, grad_log_pdf, grad_log_position)
    ...         self.new_position, self.new_momentum, self.new_grad_logp = self._get_proposed_values()
    ...     def _get_proposed_values(self):
    ...         momentum_bar = self.momentum + self.stepsize * self.grad_log_position
    ...         position_bar = self.position + self.stepsize * momentum_bar
    ...         grad_log_position, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()
    ...         return position_bar, momentum_bar, grad_log_position
    >>> pos = np.array([1, 2])
    >>> momentum = np.array([0, 0])
    >>> mean = np.array([0, 0])
    >>> covariance = np.eye(2)
    >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = ModifiedEuler(model, pos, momentum,
    ...                                                 0.25, GradLogPDFGaussian).get_proposed_values()
    >>> new_pos
    array([0.9375, 1.875])
    >>> new_momentum
    array([-0.25, -0.5])
    >>> new_grad
    array([-0.9375, -1.875])
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):
        position = _check_1d_array_object(position, "position")

        momentum = _check_1d_array_object(momentum, "momentum")

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError(
                "grad_log_pdf must be an instance"
                + " of FairnessAwareHC.inference.continuous.base.BaseGradLogPDF"
            )

        _check_length_equal(position, momentum, "position", "momentum")
        _check_length_equal(position, model.variables, "position", "model.variables")

        if grad_log_position is None:
            grad_log_position, _ = grad_log_pdf(position, model).get_gradient_log_pdf()

        else:
            grad_log_position = _check_1d_array_object(
                grad_log_position, "grad_log_position"
            )
            _check_length_equal(
                grad_log_position, position, "grad_log_position", "position"
            )

        self.position = position
        self.momentum = momentum
        self.stepsize = stepsize
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.grad_log_position = grad_log_position

        # new_position is the new proposed position, new_momentum is the new proposed momentum, new_grad_lop
        # is the value of grad log at new_position
        self.new_position = self.new_momentum = self.new_grad_logp = None

    def get_proposed_values(self):
        """
        Returns the new proposed values of position and momentum

        Returns
        -------
        Returns a tuple of following type (in order)

        numpy.array: A 1d numpy.array representing new proposed value of position

        numpy.array: A 1d numpy.array representing new proposed value of momentum

        numpy.array: A 1d numpy.array representing gradient of log distribution at new proposed value of position

        Example
        -------
        >>> # Using implementation of ModifiedEuler
        >>> from FairnessAwareHC.inference.continuous import ModifiedEuler, GradLogPDFGaussian as GLPG
        >>> from FairnessAwareHC.factors import GaussianDistribution
        >>> import numpy as np
        >>> pos = np.array([3, 4])
        >>> momentum = np.array([1, 1])
        >>> mean = np.array([-1, 1])
        >>> covariance = 3*np.eye(2)
        >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
        >>> new_pos, new_momentum, new_grad = ModifiedEuler(model, pos, momentum, 0.70, GLPG).get_proposed_values()
        >>> new_pos
        array([ 3.04666667,  4.21      ])
        >>> new_momentum
        array([ 0.06666667,  0.3       ])
        >>> new_grad
        array([-1.34888889, -1.07      ])
        """
        return self.new_position, self.new_momentum, self.new_grad_logp


class LeapFrog(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating hamiltonian dynamics using leapfrog method
    Inherits FairnessAwareHC.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    model : An instance of FairnessAwareHC.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf : A subclass of FairnessAwareHC.inference.continuous.BaseGradLogPDF, defaults to None
        A class for evaluating gradient log and log of distribution for a given assignment of variables
        If None, then model.get_gradient_log_pdf will be used

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Example
    --------
    >>> from FairnessAwareHC.factors.continuous import GaussianDistribution
    >>> from FairnessAwareHC.sampling import LeapFrog, GradLogPDFGaussian as GLPG
    >>> import numpy as np
    >>> pos = np.array([2, 1])
    >>> momentum = np.array([7, 7])
    >>> mean = np.array([-5, 5])
    >>> covariance = np.array([[1, 2], [2, 1]])
    >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = LeapFrog(model, pos, momentum, 4.0, GLPG).get_proposed_values()
    >>> new_pos
    array([ 70., -19.])
    >>> new_momentum
    array([  99., -121.])
    >>> new_grad
    array([ 41., -58.])
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):
        BaseSimulateHamiltonianDynamics.__init__(
            self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position
        )

        (
            self.new_position,
            self.new_momentum,
            self.new_grad_logp,
        ) = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using leapfrog
        """
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.stepsize * self.grad_log_position

        # Take full step in time for updating position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.stepsize * grad_log

        return position_bar, momentum_bar, grad_log


class ModifiedEuler(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating Hamiltonian Dynamics using Modified euler method
    Inherits FairnessAwareHC.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    model: An instance of FairnessAwareHC.models
        Model for which DiscretizeTime object is initialized

    position: A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_pdf: A subclass of FairnessAwareHC.inference.continuous.BaseGradLogPDF, defaults to None
        A class for finding gradient log and log of distribution
        If None, then will use model.get_gradient_log_pdf

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Example
    --------
    >>> from FairnessAwareHC.factors.continuous import GaussianDistribution
    >>> from FairnessAwareHC.sampling import GradLogPDFGaussian, ModifiedEuler
    >>> import numpy as np
    >>> pos = np.array([2, 1])
    >>> momentum = np.array([1, 1])
    >>> mean = np.array([0, 0])
    >>> covariance = np.eye(2)
    >>> model = GaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = ModifiedEuler(model, pos, momentum,
    ...                                                 0.25, GradLogPDFGaussian).get_proposed_values()
    >>> new_pos
    array([2.125, 1.1875])
    >>> new_momentum
    array([0.5, 0.75])
    >>> new_grad
    array([-2.125, -1.1875])
    """

    def __init__(
        self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position=None
    ):
        BaseSimulateHamiltonianDynamics.__init__(
            self, model, position, momentum, stepsize, grad_log_pdf, grad_log_position
        )

        (
            self.new_position,
            self.new_momentum,
            self.new_grad_logp,
        ) = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using Modified euler method
        """
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.stepsize * self.grad_log_position

        # Take full step in time and update position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        return position_bar, momentum_bar, grad_log


def _return_samples(samples, state_names_map=None):
    """
    A utility function to return samples according to type
    """
    df = pandas.DataFrame.from_records(samples)
    if state_names_map is not None:
        for var in df.columns:
            if var != "_weight":
                df[var] = df[var].map(state_names_map[var])
    return df
